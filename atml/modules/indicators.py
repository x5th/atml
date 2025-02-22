import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def compute_ATR(df, period=14):
    df = df.copy()
    df['prev_close'] = df['Close'].shift(1)
    df['tr1'] = df['High'] - df['Low']
    df['tr2'] = (df['High'] - df['prev_close']).abs()
    df['tr3'] = (df['Low'] - df['prev_close']).abs()
    df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    atr = df['TR'].rolling(window=period, min_periods=1).mean()
    return atr

def compute_MACD(df, fast_period=12, slow_period=26, signal_period=9):
    df = df.copy()
    df['EMA_fast'] = df['Close'].ewm(span=fast_period, adjust=False).mean()
    df['EMA_slow'] = df['Close'].ewm(span=slow_period, adjust=False).mean()
    df['MACD'] = df['EMA_fast'] - df['EMA_slow']
    df['MACD_signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    return df

def compute_ADX(df, period=14):
    df = df.copy()
    df['prev_close'] = df['Close'].shift(1)
    df['up_move'] = df['High'] - df['High'].shift(1)
    df['down_move'] = df['Low'].shift(1) - df['Low']
    df['+DM'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['-DM'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    df['ATR'] = compute_ATR(df, period)
    df['+DI'] = 100 * (df['+DM'].rolling(window=period, min_periods=1).sum() / (df['ATR'] + 1e-10))
    df['-DI'] = 100 * (df['-DM'].rolling(window=period, min_periods=1).sum() / (df['ATR'] + 1e-10))
    df['DX'] = 100 * (np.abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'] + 1e-10))
    adx = df['DX'].rolling(window=period, min_periods=1).mean()
    return adx

def compute_BBands(df, window=20, num_std=2):
    df = df.copy()
    sma = df['Close'].rolling(window=window, min_periods=1).mean()
    std = df['Close'].rolling(window=window, min_periods=1).std()
    df['BB_upper'] = sma + num_std * std
    df['BB_lower'] = sma - num_std * std
    df['BB_pct'] = ((df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'] + 1e-10)).clip(-2, 2)
    return df

def compute_indicators(df, sma_short_window=10, sma_long_window=50,
                       rsi_period=14, base_rsi_upper=80, base_rsi_lower=20):
    df = df.copy()
    df['SMA_short'] = df['Close'].rolling(window=sma_short_window, min_periods=1).mean()
    df['SMA_long'] = df['Close'].rolling(window=sma_long_window, min_periods=1).mean()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=rsi_period, min_periods=1).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['ATR'] = compute_ATR(df, period=14)
    df = compute_MACD(df)
    df['vol_avg'] = df['Volume'].rolling(window=20, min_periods=1).mean()
    df['MA200'] = df['Close'].rolling(window=200, min_periods=1).mean()
    df['ADX'] = compute_ADX(df, period=14)
    df = compute_BBands(df, window=20, num_std=2)
    atr_vals = df['ATR'].bfill().values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=42).fit(atr_vals)
    labels = kmeans.labels_
    mean0 = atr_vals[labels == 0].mean()
    mean1 = atr_vals[labels == 1].mean()
    volatile_cluster = 0 if mean0 > mean1 else 1
    df['regime'] = ['volatile' if lbl == volatile_cluster else 'calm' for lbl in labels]
    df['local_rsi_upper'] = df.apply(
        lambda x: base_rsi_upper + 5 if x['regime'] == 'volatile' else base_rsi_upper,
        axis=1
    )
    df['local_rsi_lower'] = df.apply(
        lambda x: base_rsi_lower - 5 if x['regime'] == 'volatile' else base_rsi_lower,
        axis=1
    )
    df = df.ffill().bfill()
    return df
