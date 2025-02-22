import numpy as np
import pandas as pd

def calculate_metrics(trades_df):
    if trades_df.empty:
        return {
            'Total Return': np.nan,
            'Win Rate': np.nan,
            'Average Profit': np.nan,
            'Max Drawdown': np.nan,
            'Sharpe Ratio': np.nan
        }
    trades_df = trades_df.sort_values('timestamp').reset_index(drop=True)
    closed_trades = trades_df[trades_df['profit'].notnull()].copy()
    closed_trades.loc[:, 'cumulative_profit'] = closed_trades['profit'].cumsum()
    total_return = closed_trades['profit'].sum()
    win_trades = closed_trades[closed_trades['profit'] > 0]
    win_rate = len(win_trades) / len(closed_trades) if len(closed_trades) > 0 else np.nan
    avg_profit = closed_trades['profit'].mean()
    running_max = closed_trades['cumulative_profit'].cummax()
    drawdown = running_max - closed_trades['cumulative_profit']
    max_drawdown = drawdown.max()
    profit_std = closed_trades['profit'].std()
    sharpe_ratio = np.mean(closed_trades['profit']) / profit_std if profit_std > 0 else np.nan
    return {
        'Total Return': total_return,
        'Win Rate': win_rate,
        'Average Profit': avg_profit,
        'Max Drawdown': max_drawdown,
        'Sharpe Ratio': sharpe_ratio
    }

def display_summary_metrics(metrics, title="Simulation Metrics"):
    import pandas as pd
    df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
    print(f"\n{title}:")
    print(df.to_string(index=False))
