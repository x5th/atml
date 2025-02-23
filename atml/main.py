import pandas as pd
import os
import pickle
import numpy as np
from datetime import datetime, timedelta
from modules.data_fetching import aggregate_data, fetch_historical_data
from modules.ml_classifier import train_online_classifier
from modules.rl_agent import RLAgent
# Using the Numba-optimized simulation function and aliasing it for consistency.
from modules.simulation import run_simulation_enhanced_ml_rl_numba as run_simulation_enhanced_ml_rl
from modules.metrics import calculate_metrics, display_summary_metrics
from modules.plotting import plot_all_pairs
from modules.dashboard import update_model_version, generate_dashboard

# Load saved model and scaler if available.
model_file = "model.pkl"
scaler_file = "scaler.pkl"
if os.path.exists(model_file) and os.path.exists(scaler_file):
    with open(model_file, "rb") as f:
        clf = pickle.load(f)
    with open(scaler_file, "rb") as f:
        scaler = pickle.load(f)
    print("Loaded saved model and scaler.")
else:
    clf, scaler = None, None

def main():
    global clf, scaler
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    interval = '1h'
    
    max_iterations = 100
    target_sharpe_lower = 1.8
    target_sharpe_upper = 2.2

    # These will store the best model parameters (as pickled objects) and the best annualised Sharpe ratio.
    best_sharpe = -np.inf
    best_model = None
    best_scaler = None

    iteration = 0
    all_iteration_metrics = []
    
    final_overall_metrics = None
    final_per_pair_metrics = None
    final_combined_trades = None
    final_aggregated_data = None

    while iteration < max_iterations:
        print(f"\nIteration {iteration + 1}")
        today = datetime.today()
        start_date = today - timedelta(days=5*365)
        start_str = start_date.strftime("%d %b %Y")   # e.g., "25 Feb 2020"
        end_str = today.strftime("%d %b %Y")           # e.g., "23 Feb 2025"
        print(f"Backtesting period: {start_str} to {end_str}")
        
        # Aggregate data (using cache if available)
        aggregated_data = aggregate_data(symbols, interval, start_str, end_str=end_str, cache_folder="cache")
        
        # Train (or update) the ML model on the aggregated data.
        clf, scaler = train_online_classifier(clf, scaler, aggregated_data)
        print("Model trained on aggregated data.")
        
        # Save the updated model and scaler.
        with open(model_file, "wb") as f:
            pickle.dump(clf, f)
        with open(scaler_file, "wb") as f:
            pickle.dump(scaler, f)
        print("Saved model and scaler to disk.")
        
        # Initialize the RL agent.
        rl_agent = RLAgent(n_bins=10, actions=[-1, 0, 1], learning_rate=0.1, discount=0.95, epsilon=0.1)
        
        # Run simulation for each trading pair.
        all_trades = []
        per_pair_metrics_list = []
        for sym in symbols:
            print(f"\nSimulating trades for {sym} ...")
            cache_file = os.path.join("cache", f"{sym}.csv")
            pair_data = fetch_historical_data(sym, interval, start_str, end_str=end_str, cache_file=cache_file)
            trades, equity = run_simulation_enhanced_ml_rl(pair_data,
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
                                                           clf=clf, scaler=scaler,
                                                           rl_agent=rl_agent)
            if not trades.empty:
                trades['symbol'] = sym
                all_trades.append(trades)
                metrics_sym = calculate_metrics(trades)
                metrics_sym['Symbol'] = sym
                per_pair_metrics_list.append(metrics_sym)
            else:
                print(f"No trades for {sym}.")
        
        if all_trades:
            combined_trades = pd.concat(all_trades, ignore_index=True)
        else:
            combined_trades = pd.DataFrame()
        
        overall_metrics = calculate_metrics(combined_trades)
        # Compute annualised Sharpe ratio based on the closed trades over 5 years.
        closed_trades = combined_trades[combined_trades['profit'].notnull()]
        n_trades = len(closed_trades)
        raw_sharpe = overall_metrics.get("Sharpe Ratio", np.nan)
        if not np.isnan(raw_sharpe) and n_trades > 0:
            annualised_sharpe = raw_sharpe * np.sqrt(n_trades / 5.0)
        else:
            annualised_sharpe = np.nan
        overall_metrics["Annualised Sharpe Ratio"] = annualised_sharpe
        
        per_pair_metrics = pd.DataFrame(per_pair_metrics_list) if per_pair_metrics_list else pd.DataFrame()
        
        display_summary_metrics(overall_metrics, title=f"Iteration {iteration+1} Overall Portfolio Metrics (All Pairs)")
        all_iteration_metrics.append(overall_metrics.copy())
        
        # Update the best model if the current annualised Sharpe is higher.
        if not np.isnan(annualised_sharpe) and annualised_sharpe > best_sharpe:
            best_sharpe = annualised_sharpe
            best_model = pickle.dumps(clf)
            best_scaler = pickle.dumps(scaler)
            print(f"New best annualised Sharpe ratio: {annualised_sharpe:.3f}")
        else:
            # Revert to the best model if the current performance is worse.
            if best_model is not None:
                clf = pickle.loads(best_model)
                scaler = pickle.loads(best_scaler)
                print(f"Reverting to best model with annualised Sharpe ratio: {best_sharpe:.3f}")
        
        # Save final iteration's results.
        final_overall_metrics = overall_metrics
        final_per_pair_metrics = per_pair_metrics
        final_combined_trades = combined_trades
        final_aggregated_data = aggregated_data
        
        # Check if the annualised Sharpe ratio is within our target range.
        if not np.isnan(annualised_sharpe) and (target_sharpe_lower <= annualised_sharpe <= target_sharpe_upper):
            print(f"Target Annualised Sharpe Ratio achieved: {annualised_sharpe:.3f}")
            break
        
        iteration += 1

    # Generate dashboard only once at the end.
    overall_plot_filename = "ensemble_model_simulation.png"
    plot_all_pairs(final_aggregated_data, filename=overall_plot_filename)
    
    model_version = update_model_version()
    generate_dashboard(final_overall_metrics, final_per_pair_metrics, final_combined_trades, None, model_version,
                       final_aggregated_data, dashboard_filename="dashboard.html",
                       overall_plot_filename=overall_plot_filename)
    
    print("\nMetrics over Iterations:")
    for i, metrics in enumerate(all_iteration_metrics):
        print(f"Iteration {i+1}: {metrics}")
    
    print("\nNext Steps for Paper Trading:")
    print("1. Securely store your API keys (via environment variables).")
    print("2. Confirm your Binance client is set to live mode.")
    print("3. Deploy this strategy in a live paper trading environment.")
    print("4. Monitor performance and continuously update your models.")

if __name__ == "__main__":
    main()
