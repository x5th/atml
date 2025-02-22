# Ensemble Trading Strategy

A modular trading strategy that combines:
1. **Classical Technical Analysis** (SMA, RSI, MACD, Volume, ATR, ADX, Bollinger Bands).
2. **Machine Learning** (SGD Classifier).
3. **Reinforcement Learning** (Simple Q-Learning agent).

## Features

- Fetches historical data from Binance (using the `python-binance` library).
- Computes multiple indicators.
- Trains or updates an ML classifier online in streaming fashion.
- Incorporates a Q-Learning agent to refine decisions.
- Simulates trades with a `combined_signal`.
- Visualizes results & trades.
- Outputs an HTML dashboard summarizing the run.

## Installation

1. Clone this repository.
2. Install dependencies from `requirements.txt`.
3. Set your Binance API keys.
4. Run `python main.py`.

## Next Steps
- Adjust intervals, symbols, etc.
- Tweak hyperparameters.
- Extend to live trading.

## License
[MIT](./LICENSE)
