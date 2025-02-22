#!/usr/bin/env python3
"""
hyperparameter_search.py

This script performs an offline hyperparameter search for our trading strategy,
with the objective of improving the Sharpe ratio.
It uses BTCUSDT historical data (1h interval starting from '1 Jan 2023'),
splits the data into training and validation windows,
and loops over a grid of hyperparameters.
"""

import itertools
import pandas as pd
import numpy as np

# Import our modules
from modules.data_fetching import fetch_binance_data
from modules.ml_classifier import train_online_classifier
from modules.simulation import run_simulation_enhanced_ml_rl
from modules.metrics import calculate_metrics

def main():
    # ---------------------------
    # 1. Load and Split the Data
    # ---------------------------
    # For this example, we'll use BTCUSDT data.
    symbol = 'BTCUSDT'
    interval = '1h'
    start_str = '1 Jan 2023'
    
    print("Fetching data for hyperparameter search...")
    data = fetch_binance_data(symbol, interval, start_str)
    
    if data.empty:
        print("No data fetched. Check your API keys and parameters.")
        return

    # Split data: first 70% for training, last 30% for validation.
    split_index = int(0.7 * len(data))
    train_data = data.iloc[:split_index]
    val_data = data.iloc[split_index:]
    print(f"Total data points: {len(data)}; Training: {len(train_data)}; Validation: {len(val_data)}")
    
    # --------------------------------
    # 2. Define Hyperparameter Grid
    # --------------------------------
    # We will vary the classifier's alpha (regularization) and the entry/exit thresholds.
    alphas = [1e-4, 1e-3, 1e-2]
    entry_thresholds = [0.03, 0.05, 0.07]
    exit_thresholds = [-0.03, -0.05, -0.07]
    
    # We can combine parameters using itertools.product.
    param_combinations = list(itertools.product(alphas, entry_thresholds, exit_thresholds))
    print(f"Total parameter combinations: {len(param_combinations)}")
    
    # -------------------------------------
    # 3. Loop Over Hyperparameter Combinations
    # -------------------------------------
    best_sharpe = -np.inf
    best_params = None
    results = []  # to store each result for analysis

    for alpha, entry_thresh, exit_thresh in param_combinations:
        print(f"\nTesting parameters: alpha={alpha}, entry_threshold={entry_thresh}, exit_threshold={exit_thresh}")
        
        # Initialize classifier and scaler as None.
        clf = None
        scaler = None
        
        # Train the classifier on the training window.
        try:
            clf, scaler = train_online_classifier(clf, scaler, train_data)
        except Exception as e:
            print(f"Error during training: {e}")
            continue
        
        # Run simulation on the validation window using these thresholds.
        try:
            trades, equity = run_simulation_enhanced_ml_rl(
                val_data,
                entry_threshold=entry_thresh,
                exit_threshold=exit_thresh,
                clf=clf,
                scaler=scaler
                # other parameters use default values
            )
        except Exception as e:
            print(f"Error during simulation: {e}")
            continue
        
        # Calculate metrics
        metrics = calculate_metrics(trades)
        sharpe = metrics.get("Sharpe Ratio", -np.inf)
        print(f"Sharpe Ratio for parameters (alpha={alpha}, entry={entry_thresh}, exit={exit_thresh}): {sharpe}")
        
        results.append({
            "alpha": alpha,
            "entry_threshold": entry_thresh,
            "exit_threshold": exit_thresh,
            "Sharpe Ratio": sharpe
        })
        
        if sharpe is not None and sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = (alpha, entry_thresh, exit_thresh)
    
    # Convert results to DataFrame for easier analysis.
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Sharpe Ratio", ascending=False)
    
    print("\nHyperparameter Search Results:")
    print(results_df)
    
    if best_params:
        print(f"\nBest parameters found: alpha={best_params[0]}, entry_threshold={best_params[1]}, exit_threshold={best_params[2]}")
        print(f"Best Sharpe Ratio: {best_sharpe}")
    else:
        print("No valid parameter combination found.")

if __name__ == "__main__":
    main()

