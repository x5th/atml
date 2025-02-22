import pandas as pd
import numpy as np
from modules.config import get_binance_client
from modules.data_fetching import aggregate_data, fetch_binance_data
from modules.ml_classifier import train_online_classifier
from modules.rl_agent import RLAgent
from modules.simulation import run_simulation_enhanced_ml_rl
from modules.metrics import calculate_metrics, display_summary_metrics
from modules.plotting import plot_all_pairs
from modules.dashboard import update_model_version, generate_dashboard

def main():
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    interval = '1h'
    start_str = '1 Jan 2023'
    aggregated_data = aggregate_data(symbols, interval, start_str)
    clf, scaler = None, None
    clf, scaler = train_online_classifier(clf, scaler, aggregated_data)
    rl_agent = RLAgent(n_bins=10, actions=[-1, 0, 1], learning_rate=0.1, discount=0.95, epsilon=0.1)
    all_trades = []
    per_pair_metrics_list = []
    for sym in symbols:
        print(f"\nSimulating trades for {sym} ...")
        pair_data = fetch_binance_data(sym, interval, start_str)
        trades, final_equity = run_simulation_enhanced_ml_rl(
            pair_data,
            sma_short_window=10,
            sma_long_window=50,
            base_rsi_upper=80,
            base_rsi_lower=20,
            risk_multiplier=1.5,
            initial_capital=10000,
            risk_fraction=0.01,
            w_sma=1/6, w_rsi=1/6, w_macd=1/6, w_vol=1/6,
            w_ml=1/6, w_rl=1/6,
            entry_threshold=0.05,
            exit_threshold=-0.05,
            clf=clf,
            scaler=scaler,
            rl_agent=rl_agent
        )
        if not trades.empty:
            trades['symbol'] = sym
            all_trades.append(trades)
            metrics_sym = calculate_metrics(trades)
            metrics_sym['Symbol'] = sym
            per_pair_metrics_list.append(metrics_sym)
        else:
            print(f"No trades for {sym}")
    if all_trades:
        combined_trades = pd.concat(all_trades, ignore_index=True)
    else:
        combined_trades = pd.DataFrame()
    overall_metrics = calculate_metrics(combined_trades)
    if per_pair_metrics_list:
        per_pair_metrics = pd.DataFrame(per_pair_metrics_list)
    else:
        per_pair_metrics = pd.DataFrame()
    display_summary_metrics(overall_metrics, title="Overall Portfolio Metrics (All Pairs)")
    test_data = fetch_binance_data('BTCUSDT', interval, start_str)
    window_size = 50
    step_size = 25
    metrics_list = []
    start_idx = 0
    while start_idx + window_size < len(test_data):
        window_data = test_data.iloc[start_idx : start_idx + window_size]
        clf, scaler = train_online_classifier(clf, scaler, window_data)
        trades_window, equity_window = run_simulation_enhanced_ml_rl(
            window_data,
            sma_short_window=10,
            sma_long_window=50,
            base_rsi_upper=80,
            base_rsi_lower=20,
            risk_multiplier=1.5,
            initial_capital=10000,
            risk_fraction=0.01,
            w_sma=1/6, w_rsi=1/6, w_macd=1/6, w_vol=1/6,
            w_ml=1/6, w_rl=1/6,
            entry_threshold=0.05,
            exit_threshold=-0.05,
            clf=clf,
            scaler=scaler,
            rl_agent=rl_agent
        )
        m = calculate_metrics(trades_window)
        m['window_start'] = window_data.index[0]
        m['window_end'] = window_data.index[-1]
        metrics_list.append(m)
        start_idx += step_size
    wfa_results = pd.DataFrame(metrics_list)
    print("\nWalk-Forward Analysis Results (BTCUSDT Example):")
    print(wfa_results)
    wfa_results.to_csv("enhanced_ml_rl_walk_forward_results.csv", index=False)
    plot_all_pairs(aggregated_data, filename="ensemble_model_simulation.png")
    model_version = update_model_version()
    generate_dashboard(
        overall_metrics,
        per_pair_metrics,
        combined_trades,
        wfa_results,
        model_version,
        aggregated_data,
        dashboard_filename="dashboard.html",
        overall_plot_filename="ensemble_model_simulation.png"
    )
    print("\nNext Steps for Paper Trading:")
    print("1. Securely store and use your API keys (via environment variables).")
    print("2. Confirm your Binance client is set to testnet.")
    print("3. Deploy this ensemble strategy in a live paper trading environment.")
    print("4. Monitor performance, trade logs, and continuously update your models.")
    print("5. Experiment with additional data sources and model architectures for further improvements.")

if __name__ == "__main__":
    main()
