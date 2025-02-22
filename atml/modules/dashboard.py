import os
import pandas as pd
from .plotting import plot_performance_for_each_pair

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

def generate_dashboard(overall_metrics,
                       per_pair_metrics,
                       trades_df,
                       wfa_results,
                       model_version,
                       aggregated_data,
                       dashboard_filename="dashboard.html",
                       overall_plot_filename="ensemble_model_simulation.png"):
    symbol_plots = plot_performance_for_each_pair(aggregated_data, trades_df)
    css_block = (
        "<style>\n"
        "body {font-family: Arial, sans-serif; margin: 20px; background-color: #f9f9f9;}\n"
        "h1, h2, h3 {color: #333;}\n"
        "table {border-collapse: collapse; width: 100%; margin-bottom: 20px; background-color: #fff;}\n"
        "th, td {border: 1px solid #ddd; padding: 8px; text-align: center;}\n"
        "th {background-color: #f2f2f2;}\n"
        ".section {margin-bottom: 40px;}\n"
        "</style>\n"
    )
    html = "<html><head><title>Ensemble Trading Dashboard</title>" + css_block + "</head><body>"
    html += f"<h1>Ensemble Trading Dashboard</h1>"
    html += f"<h2>Model Version: {model_version}</h2>"
    html += '<div class="section"><h2>Overall Summary Metrics</h2>'
    html += pd.DataFrame(overall_metrics.items(), columns=["Metric", "Value"]).to_html(index=False)
    html += "</div>"
    html += '<div class="section"><h2>Per-Pair Summary Metrics</h2>'
    if not per_pair_metrics.empty:
        html += per_pair_metrics.to_html(index=False)
    else:
        html += "<p>No per-pair metrics found.</p>"
    html += "</div>"
    html += '<div class="section"><h2>Trades (All Pairs)</h2>'
    if not trades_df.empty:
        html += trades_df.to_html(index=False)
    else:
        html += "<p>No trades recorded.</p>"
    html += "</div>"
    if wfa_results is not None and not wfa_results.empty:
        html += '<div class="section"><h2>Walk-Forward Analysis</h2>'
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
    html += "</body></html>"
    with open(dashboard_filename, "w") as f:
        f.write(html)
    print(f"Dashboard generated: {dashboard_filename}")
