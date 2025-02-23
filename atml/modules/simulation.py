import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from .indicators import compute_indicators
from numba import njit

@njit
def simulate_trades_numba(close, atr, combined_signal, entry_threshold, exit_threshold, risk_multiplier, initial_capital, risk_fraction):
    n = close.shape[0]
    equity = initial_capital
    position = 0  # 0: no position, 1: LONG
    entry_price = 0.0
    position_size = 0.0
    stop_loss = 0.0
    max_price = 0.0

    # Preallocate arrays for trade events (maximum n events)
    trade_times = np.empty(n, dtype=np.int64)
    trade_actions = np.empty(n, dtype=np.int64)  # 1 for BUY, -1 for SELL
    trade_prices = np.empty(n, dtype=np.float64)
    trade_equity = np.empty(n, dtype=np.float64)
    trade_profits = np.empty(n, dtype=np.float64)
    trade_count = 0

    for i in range(n):
        price = close[i]
        current_atr = atr[i]
        sig = combined_signal[i]
        if position == 0 and sig > entry_threshold:
            risk_amount = equity * risk_fraction
            stop_loss = price - risk_multiplier * current_atr
            if price - stop_loss <= 0:
                continue
            position_size = risk_amount / (price - stop_loss)
            position = 1
            entry_price = price
            max_price = price
            trade_times[trade_count] = i
            trade_actions[trade_count] = 1  # BUY
            trade_prices[trade_count] = price
            trade_equity[trade_count] = equity
            trade_profits[trade_count] = 0.0
            trade_count += 1
        elif position == 1:
            if price > max_price:
                max_price = price
                new_stop = max_price - risk_multiplier * current_atr
                if new_stop > stop_loss:
                    stop_loss = new_stop
            if sig < exit_threshold or price <= stop_loss:
                exit_price = price
                profit = (exit_price - entry_price) * position_size
                equity += profit
                trade_times[trade_count] = i
                trade_actions[trade_count] = -1  # SELL
                trade_prices[trade_count] = exit_price
                trade_equity[trade_count] = equity
                trade_profits[trade_count] = profit
                trade_count += 1
                position = 0
                entry_price = 0.0
                position_size = 0.0
                stop_loss = 0.0
                max_price = 0.0
    # Slice the trade arrays to the number of trades recorded.
    trade_times = trade_times[:trade_count]
    trade_actions = trade_actions[:trade_count]
    trade_prices = trade_prices[:trade_count]
    trade_equity = trade_equity[:trade_count]
    trade_profits = trade_profits[:trade_count]
    return equity, trade_times, trade_actions, trade_prices, trade_equity, trade_profits

def run_simulation_enhanced_ml_rl_numba(
        data,
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
        clf=None, scaler=None,
        rl_agent=None
    ):
    # Compute indicators and signals as before.
    df = compute_indicators(data, sma_short_window, sma_long_window, 14, base_rsi_upper, base_rsi_lower)
    df['sma_signal'] = np.where(df['SMA_short'] > df['SMA_long'], 1, -1)
    epsilon = 1e-10
    denom = (df['local_rsi_upper'] - df['local_rsi_lower']).replace(0, np.nan)
    df['rsi_signal'] = 2 * (df['local_rsi_upper'] - df['RSI']) / (denom + epsilon) - 1
    df['rsi_signal'] = df['rsi_signal'].clip(-2, 2).fillna(0)
    df['macd_signal'] = np.where(df['MACD_hist'] > 0, 1, -1)
    df['vol_signal'] = np.where(df['Volume'] >= df['vol_avg'], 1, -1)
    # For faster simulation, we set ML and RL signals to zero.
    df['ml_signal'] = 0
    df['rl_signal'] = 0
    df['combined_signal'] = (
        w_sma * df['sma_signal'] +
        w_rsi * df['rsi_signal'] +
        w_macd * df['macd_signal'] +
        w_vol * df['vol_signal'] +
        w_ml  * df['ml_signal'] +
        w_rl  * df['rl_signal']
    )
    df['combined_signal'] = df['combined_signal'].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-10, 10)
    
    print("Combined Signal Statistics:")
    print(df['combined_signal'].describe())
    print("\nLast 10 rows of signals:")
    print(df[['sma_signal','rsi_signal','macd_signal','vol_signal','ml_signal','rl_signal','combined_signal']].tail(10))
    
    # Extract arrays for simulation.
    close = df['Close'].values
    atr = df['ATR'].values
    combined_signal = df['combined_signal'].values
    
    # Call the Numba-accelerated simulation loop.
    equity, trade_times, trade_actions, trade_prices, trade_equity, trade_profits = simulate_trades_numba(
        close, atr, combined_signal, entry_threshold, exit_threshold, risk_multiplier, initial_capital, risk_fraction
    )
    
    # Reconstruct a DataFrame for trades (using index positions as a proxy for timestamps)
    trades = {
        'timestamp': df.index[trade_times],
        'action': ['BUY' if a == 1 else 'SELL' for a in trade_actions],
        'price': trade_prices,
        'equity': trade_equity,
        'profit': trade_profits
    }
    trades_df = pd.DataFrame(trades)
    return trades_df, equity
