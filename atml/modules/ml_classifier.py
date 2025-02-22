import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
from .indicators import compute_indicators

def generate_ml_features(df):
    df = df.copy()
    df['sma_spread'] = (df['SMA_short'] - df['SMA_long']) / (df['SMA_long'] + 1e-10)
    df['vol_ratio'] = (df['Volume'] + 1e-10) / (df['vol_avg'] + 1e-10)
    df['next_return'] = df['Close'].shift(-1) / df['Close'] - 1
    df['target'] = (df['next_return'] > 0.001).astype(int)
    df = df[['sma_spread', 'RSI', 'MACD_hist', 'vol_ratio', 'ADX', 'BB_pct', 'target']].dropna()
    features = df[['sma_spread', 'RSI', 'MACD_hist', 'vol_ratio', 'ADX', 'BB_pct']]
    target = df['target']
    return features, target

def train_online_classifier(clf, scaler, new_data,
                            sma_short_window=10, sma_long_window=50,
                            base_rsi_upper=80, base_rsi_lower=20):
    new_data = compute_indicators(new_data, sma_short_window, sma_long_window,
                                  rsi_period=14,
                                  base_rsi_upper=base_rsi_upper,
                                  base_rsi_lower=base_rsi_lower)
    features, target = generate_ml_features(new_data)
    X = features.ffill().bfill().fillna(0).values
    y = target.values
    if clf is None:
        clf = SGDClassifier(loss='log_loss', random_state=42, alpha=1e-4)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        clf.partial_fit(X_scaled, y, classes=np.array([0, 1]))
    else:
        X_scaled = scaler.transform(X)
        clf.partial_fit(X_scaled, y)
    return clf, scaler
