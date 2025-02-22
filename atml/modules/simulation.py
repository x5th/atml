import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from .indicators import compute_indicators

def run_simulation_enhanced_ml_rl(
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
    df = compute_indicators(data, sma_short_window, sma_long_window, 14, base_rsi_upper, base_rsi_lower)
    df['sma_signal'] = np.where(df['SMA_short'] > df['SMA_long'], 1, -1)
    epsilon = 1e-10
    denom = (df['local_rsi_upper'] - df['local_rsi_lower']).replace(0, np.nan)
    df['rsi_signal'] = 2 * (df['local_rsi_upper'] - df['RSI']) / (denom + epsilon) - 1
    df['rsi_signal'] = df['rsi_signal'].clip(-2, 2).fillna(0)
    df['macd_signal'] = np.where(df['MACD_hist'] > 0, 1, -1)
    df['vol_signal'] = np.where(df['Volume'] >= df['vol_avg'], 1, -1)
    if clf is not None and scaler is not None:
        df_ml = df[['SMA_short','SMA_long','RSI','MACD_hist','Volume','vol_avg','ADX','BB_pct']].copy()
        df_ml['sma_spread'] = (df_ml['SMA_short'] - df_ml['SMA_long']) / (df_ml['SMA_long'] + epsilon)
        df_ml['vol_ratio'] = (df_ml['Volume'] + epsilon) / (df_ml['vol_avg'] + epsilon)
        ml_features = df_ml[['sma_spread', 'RSI', 'MACD_hist', 'vol_ratio', 'ADX', 'BB_pct']].ffill().bfill().fillna(0)
        try:
            prob = clf.predict_proba(scaler.transform(ml_features.values))[:, 1]
            df['ml_signal'] = (2 * prob - 1).clip(-2, 2)
        except NotFittedError:
            df['ml_signal'] = 0
    else:
        df['ml_signal'] = 0
    df['rl_signal'] = 0
    if rl_agent is not None:
        for idx, row in df.iterrows():
            state_vector = [row['RSI'], row['MACD_hist']]
            action = rl_agent.choose_action(state_vector)
            df.at[idx, 'rl_signal'] = action
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
    equity = initial_capital
    position = None
    entry_price = None
    position_size = 0
    stop_loss = None
    max_price = None
    trades = []
    prev_state = None
    for timestamp, row in df.iterrows():
        price = row['Close']
        atr = row['ATR']
        combined = row['combined_signal']
        current_state = [row['RSI'], row['MACD_hist']]
        if position is None and combined > entry_threshold:
            risk_amount = equity * risk_fraction
            stop_loss = price - risk_multiplier * atr
            if price - stop_loss <= 0:
                continue
            position_size = risk_amount / (price - stop_loss)
            position = 'LONG'
            entry_price = price
            max_price = price
            trades.append({'timestamp': timestamp,'action': 'BUY','price': price,'position_size': position_size,'equity': equity})
            prev_state = current_state
        elif position == 'LONG':
            if price > max_price:
                max_price = price
                new_stop = max_price - risk_multiplier * atr
                stop_loss = max(stop_loss, new_stop)
            if (combined < exit_threshold) or (price <= stop_loss):
                exit_price = price
                profit = (exit_price - entry_price) * position_size
                equity += profit
                trades.append({'timestamp': timestamp,'action': 'SELL','price': exit_price,'profit': profit,'equity': equity,'exit_reason': 'Signal/Stop'})
                if rl_agent is not None and prev_state is not None:
                    reward = profit / (price * 0.01)
                    rl_agent.update(prev_state, 1, reward, current_state)
                position = None
                entry_price = None
                position_size = 0
                stop_loss = None
                max_price = None
                prev_state = None
    trades_df = pd.DataFrame(trades)
    return trades_df, equity
