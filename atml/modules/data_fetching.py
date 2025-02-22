import logging
import pandas as pd
from .config import get_binance_client

client = get_binance_client()

def fetch_binance_data(symbol, interval, start_str):
    """Fetches historical klines for a given symbol, interval, and start date."""
    try:
        klines = client.get_historical_klines(symbol, interval, start_str)
        cols = [
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close time', 'Quote asset volume', 'Number of trades',
            'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
        ]
        data = pd.DataFrame(klines, columns=cols)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            data[col] = data[col].astype(float)
        data['Open time'] = pd.to_datetime(data['Open time'], unit='ms')
        data.set_index('Open time', inplace=True)
        data['symbol'] = symbol
        return data
    except Exception as e:
        logging.exception(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def aggregate_data(symbols, interval, start_str):
    """Aggregates fetched data for a list of symbols into a single DataFrame."""
    data_list = []
    for sym in symbols:
        df = fetch_binance_data(sym, interval, start_str)
        if not df.empty:
            data_list.append(df)
    if data_list:
        aggregated = pd.concat(data_list)
        aggregated.sort_index(inplace=True)
        return aggregated
    else:
        return pd.DataFrame()
