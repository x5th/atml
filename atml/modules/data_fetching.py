import os
import pandas as pd
import time
from datetime import datetime
from modules.config import get_binance_client

client = get_binance_client()

def fetch_historical_data(symbol, interval, start_str, end_str=None, cache_file=None):
    """
    Fetch historical kline data for a given symbol using pagination.
    If a cache_file is provided and exists, load the data from that file.
    
    Args:
      symbol (str): e.g., "BTCUSDT"
      interval (str): e.g., "1h"
      start_str (str): Start date as a string (e.g., "22 Feb 2018").
      end_str (str, optional): End date as a string (e.g., "22 Feb 2023").
      cache_file (str, optional): Filename to use for caching data.
      
    Returns:
      pd.DataFrame: Historical OHLCV data with a 'symbol' column.
    """
    # If cache_file is specified and exists, load the cached data.
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached data for {symbol} from {cache_file}")
        data = pd.read_csv(cache_file, index_col="Open time", parse_dates=True)
        return data

    if end_str is None:
        end_str = datetime.now().strftime("%d %b %Y")
    
    all_klines = []
    limit = 1000  # Maximum candles per request
    
    print(f"Fetching data for {symbol} from {start_str} to {end_str}")
    klines = client.get_historical_klines(symbol, interval, start_str, end_str=end_str, limit=limit)
    iteration = 1
    while klines:
        print(f"Iteration {iteration}: Fetched {len(klines)} candles")
        all_klines.extend(klines)
        if len(klines) < limit:
            break
        last_open_time = klines[-1][0]
        new_start = pd.to_datetime(last_open_time, unit='ms') + pd.Timedelta(seconds=1)
        new_start_str = new_start.strftime("%d %b %Y %H:%M:%S")
        time.sleep(0.5)  # Delay to avoid triggering WAF limits
        klines = client.get_historical_klines(symbol, interval, new_start_str, end_str=end_str, limit=limit)
        iteration += 1
    
    cols = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 
            'Close time', 'Quote asset volume', 'Number of trades', 
            'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
    data = pd.DataFrame(all_klines, columns=cols)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        data[col] = data[col].astype(float)
    data['Open time'] = pd.to_datetime(data['Open time'], unit='ms')
    data.set_index('Open time', inplace=True)
    data['symbol'] = symbol

    if not data.empty:
        print(f"Data range for {symbol}: {data.index[0]} to {data.index[-1]}")
    else:
        print(f"No data fetched for {symbol}")
    
    # Save to cache file if provided.
    if cache_file:
        print(f"Saving data for {symbol} to {cache_file}")
        data.to_csv(cache_file)
    
    return data

def aggregate_data(symbols, interval, start_str, end_str=None, cache_folder="cache"):
    """
    Aggregate historical data for multiple symbols.
    Each symbol's data will be cached in a separate CSV file.
    
    Args:
      symbols (list): e.g., ['BTCUSDT', 'ETHUSDT']
      interval (str): e.g., "1h"
      start_str (str): Start date string.
      end_str (str, optional): End date string.
      cache_folder (str): Folder where cache files will be stored.
      
    Returns:
      pd.DataFrame: Aggregated historical data for all symbols.
    """
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)
    data_list = []
    for sym in symbols:
        cache_file = os.path.join(cache_folder, f"{sym}.csv")
        try:
            df = fetch_historical_data(sym, interval, start_str, end_str=end_str, cache_file=cache_file)
            data_list.append(df)
        except Exception as e:
            print(f"Error fetching data for {sym}: {e}")
    if data_list:
        aggregated = pd.concat(data_list)
        aggregated.sort_index(inplace=True)
        return aggregated
    else:
        return pd.DataFrame()
