# optimization/hyperparams.py
import json
import numpy as np
import pandas as pd
from ml.ml import train_online_classifier
from simulation.simulation import run_simulation, calculate_metrics
from data.data import aggregate_data
from datetime import datetime, timedelta

def grid_search_parameters(aggregated_data, symbols, interval, base_rsi_upper=80, base_rsi_lower=20):
    # Define a grid for some example parameters (adjust ranges as needed)
    sma_short_options = [5, 10, 15]
    sma_long_options = [30, 50, 70]
    risk_multiplier_options = [1.0, 1.5, 2.0]
    entry_threshold_options = [0.03, 0.05, 0.07]
    exit_threshold_options = [-0.05, -0.07, -0.10]

    best_score = -np.inf
    best_params = {}
    # Use BTCUSDT for evaluation in this simple example
    data = aggregated_data[aggregated_data['symbol'] == 'BTCUSDT']
    # Use a subset for faster evaluation (e.g., last 1 year)
    one_year_ago = pd.Timestamp.today() - pd.DateOffset(years=1)
    data = data[data.index >= one_year_ago]

    for sma_short in sma_short_options:
        for sma_long in sma_long_options:
            for risk_multiplier in risk_multiplier_options:
                for entry_threshold in entry_threshold_options:
                    for exit_threshold in exit_threshold_options:
                        # Use a fresh classifier and scaler
                        clf, scaler = None, None
                        clf, scaler = train_online_classifier(clf, scaler, data,
                                                                sma_short_window=sma_short,
                                                                sma_long_window=sma_long,
                                                                base_rsi_upper=base_rsi_upper,
                                                                base_rsi_lower=base_rsi_lower)
                        # Run simulation on the subset data
                        trades, _ = run_simulation(data, clf, scaler, None,
                                                     sma_short_window=sma_short,
                                                     sma_long_window=sma_long,
                                                     base_rsi_upper=base_rsi_upper,
                                                     base_rsi_lower=base_rsi_lower,
                                                     risk_multiplier=risk_multiplier,
                                                     initial_capital=10000,
                                                     risk_fraction=0.01,
                                                     entry_threshold=entry_threshold,
                                                     exit_threshold=exit_threshold)
                        metrics = calculate_metrics(trades)
                        # Use total return (or Sharpe Ratio) as score; here, total return is used.
                        score = metrics.get('Total Return', -np.inf)
                        if score > best_score:
                            best_score = score
                            best_params = {
                                'sma_short_window': sma_short,
                                'sma_long_window': sma_long,
                                'risk_multiplier': risk_multiplier,
                                'entry_threshold': entry_threshold,
                                'exit_threshold': exit_threshold
                            }
    return best_params, best_score

def load_latest_parameters(params_file="best_params.json"):
    try:
        with open(params_file, "r") as f:
            params = json.load(f)
        return params
    except FileNotFoundError:
        return None

def save_latest_parameters(params, params_file="best_params.json"):
    with open(params_file, "w") as f:
        json.dump(params, f)
