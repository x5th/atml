import os
import pandas as pd
import json
from modules.plotting import plot_performance_for_each_pair
from modules.indicators import compute_indicators

def update_model_version(version_file="model_version.txt"):
    try:
        with open(version_file, "r") as f:
            version = f.read().strip()
            parts = version.split(".")
            parts[-1] = str(int(parts[-1]) + 1)
            new_version = ".".join(parts)
    except FileNotFoundError:
        new_version = "1.0.0"
    with open(version_file, "w") as f:
        f.write(new_version)
    return new_version

def trades_per_month(trades_df):
    if trades_df.empty:
        return pd.DataFrame()
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    trades_df['Month'] = trades_df['timestamp'].dt.to_period('M')
    monthly_counts = trades_df.groupby(['symbol', 'Month']).size().reset_index(name='Trade Count')
    return monthly_counts

def generate_dashboard(overall_metrics, per_pair_metrics, trades_df, wfa_results, model_version,
                       aggregated_data, dashboard_filename="dashboard.html",
                       overall_plot_filename="ensemble_model_simulation.png"):
    # Plot individual pair performance diagrams.
    symbol_plots = plot_performance_for_each_pair(aggregated_data, trades_df)
    monthly_counts = trades_per_month(trades_df)
    
    # --- Signals Section ---
    signals_df = compute_indicators(aggregated_data.copy(), sma_short_window=10, sma_long_window=50,
                                    rsi_period=14, base_rsi_upper=80, base_rsi_lower=20)
    signals_df['sma_signal'] = (signals_df['SMA_short'] > signals_df['SMA_long']).astype(int).replace({0: -1})
    epsilon = 1e-10
    denom = (signals_df['local_rsi_upper'] - signals_df['local_rsi_lower']).replace(0, pd.NA)
    signals_df['rsi_signal'] = 2 * (signals_df['local_rsi_upper'] - signals_df['RSI']) / (denom + epsilon) - 1
    signals_df['rsi_signal'] = signals_df['rsi_signal'].clip(-2, 2).fillna(0)
    signals_df['macd_signal'] = (signals_df['MACD_hist'] > 0).astype(int).replace({0: -1})
    signals_df['vol_signal'] = (signals_df['Volume'] >= signals_df['vol_avg']).astype(int).replace({0: -1})
    signals_df['ml_signal'] = 0
    signals_df['rl_signal'] = 0
    signals_df['combined_signal'] = (1/6)*signals_df['sma_signal'] + (1/6)*signals_df['rsi_signal'] + \
                                    (1/6)*signals_df['macd_signal'] + (1/6)*signals_df['vol_signal'] + \
                                    (1/6)*signals_df['ml_signal'] + (1/6)*signals_df['rl_signal']
    latest_signals = signals_df.tail(10).reset_index()
    
    # --- Model Improvement Section ---
    prev_metrics_file = "prev_metrics.json"
    improvement_html = "<h2>Model Improvement from Last Run</h2>"
    if os.path.exists(prev_metrics_file):
        with open(prev_metrics_file, "r") as f:
            prev_metrics = json.load(f)
        diff = {}
        for key, value in overall_metrics.items():
            prev = prev_metrics.get(key, None)
            # Only calculate difference if both current and previous are numeric.
            if isinstance(value, (int, float)) and isinstance(prev, (int, float)):
                diff[key] = value - prev
            else:
                diff[key] = "N/A"
        improvement_html += pd.DataFrame(list(diff.items()), columns=["Metric", "Difference"]).to_html(index=False)
    else:
        improvement_html += "<p>No previous metrics available for comparison.</p>"
    with open(prev_metrics_file, "w") as f:
        json.dump(overall_metrics, f)
    
    # --- Construct Dashboard HTML ---
    html = "<html><head><title>Ensemble Trading Dashboard</title>"
    html += """
    <style>
    body {font-family: Arial, sans-serif; margin: 20px; background-color: #f9f9f9;}
    h1, h2, h3 {color: #333;}
    table {border-collapse: collapse; width: 100%; margin-bottom: 20px; background-color: #fff;}
    th, td {border: 1px solid #ddd; padding: 8px; text-align: center;}
    th {background-color: #f2f2f2;}
    .section {margin-bottom: 40px;}
    </style>
    """
    html += "</head><body>"
    html += f"<h1>Ensemble Trading Dashboard</h1>"
    html += f"<h2>Model Version: {model_version}</h2>"
    
    html += '<div class="section"><h2>Overall Summary Metrics</h2>'
    html += pd.DataFrame(list(overall_metrics.items()), columns=["Metric", "Value"]).to_html(index=False)
    html += "</div>"
    
    html += '<div class="section"><h2>Per-Pair Summary Metrics</h2>'
    html += per_pair_metrics.to_html(index=False)
    html += "</div>"
    
    html += '<div class="section"><h2>Monthly Trade Counts</h2>'
    if not monthly_counts.empty:
        html += monthly_counts.to_html(index=False)
    else:
        html += "<p>No monthly trade data available.</p>"
    html += "</div>"
    
    if wfa_results is not None and not wfa_results.empty:
        html += '<div class="section"><h2>Walk-Forward Analysis (BTCUSDT Example)</h2>'
        html += wfa_results.to_html(index=False)
        html += "</div>"
    
    html += '<div class="section"><h2>Aggregated Price Simulation (All Pairs)</h2>'
    html += f'<img src="{overall_plot_filename}" alt="Aggregated Price Simulation" width="100%"/>'
    html += "</div>"
    
    html += '<div class="section"><h2>Individual Pair Performance Diagrams</h2>'
    for sym, plot_file in symbol_plots.items():
         html += f"<h3>{sym} Performance Diagram</h3>"
         html += f'<img src="{plot_file}" alt="{sym} Performance" width="100%"/><br/>'
    html += "</div>"
    
    html += '<div class="section"><h2>Latest Signals (Last 10 Rows)</h2>'
    html += latest_signals.to_html(index=False)
    html += "</div>"
    
    html += f'<div class="section">{improvement_html}</div>'
    
    html += "</body></html>"
    with open(dashboard_filename, "w") as f:
        f.write(html)
    print(f"Dashboard generated: {dashboard_filename}")
