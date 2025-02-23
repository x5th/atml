import numpy as np
import pandas as pd

def calculate_metrics(trades_df, initial_capital=10000):
    if trades_df.empty:
        return {
            'Total Return': np.nan,
            'Win Rate': np.nan,
            'Average Profit': np.nan,
            'Max Drawdown': np.nan,
            'Max Drawup': np.nan,
            'Raw Sharpe Ratio': np.nan,
            'Annualised Sharpe Ratio': np.nan,
            'Sharpe Interpretation': "N/A"
        }
    # Ensure trades are sorted and timestamps are in datetime format
    trades_df = trades_df.sort_values('timestamp').reset_index(drop=True)
    closed_trades = trades_df[trades_df['profit'].notnull()].copy()
    
    # Create a cumulative profit series
    closed_trades.loc[:, 'cumulative_profit'] = closed_trades['profit'].cumsum()
    total_return = closed_trades['profit'].sum()
    win_trades = closed_trades[closed_trades['profit'] > 0]
    win_rate = len(win_trades) / len(closed_trades) if len(closed_trades) > 0 else np.nan
    avg_profit = closed_trades['profit'].mean()
    running_max = closed_trades['cumulative_profit'].cummax()
    drawdown = running_max - closed_trades['cumulative_profit']
    max_drawdown = drawdown.max()
    running_min = closed_trades['cumulative_profit'].cummin()
    drawup = closed_trades['cumulative_profit'] - running_min
    max_drawup = drawup.max()
    
    # Compute trade returns (as a fraction of initial capital)
    returns = closed_trades['profit'] / initial_capital

    # Raw Sharpe: using trade returns (non-annualised)
    raw_sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else np.nan

    # Determine the simulation period (in years) using the trade timestamps.
    # Ensure that the 'timestamp' column is in datetime format.
    closed_trades['timestamp'] = pd.to_datetime(closed_trades['timestamp'])
    T_years = (closed_trades['timestamp'].iloc[-1] - closed_trades['timestamp'].iloc[0]).days / 365.0
    N = len(closed_trades)
    
    # Annualise the trade returns:
    # Assume that, on average, the total annual return is approximately:
    #   annual_return = (mean trade return) * (number of trades per year),
    # and the volatility scales as sqrt(number of trades per year)
    trades_per_year = N / T_years if T_years > 0 else np.nan
    annual_return = np.mean(returns) * trades_per_year if trades_per_year is not np.nan else np.nan
    annual_vol = np.std(returns) * np.sqrt(trades_per_year) if trades_per_year is not np.nan and np.std(returns)>0 else np.nan
    
    # Now subtract the risk free rate (assumed 5% annual)
    annual_sharpe = (annual_return - 0.05) / annual_vol if annual_vol > 0 else np.nan

    # Provide a simple interpretation
    def interpret_sharpe(sharpe):
        if np.isnan(sharpe):
            return "N/A"
        elif sharpe < 1:
            return "Poor"
        elif sharpe < 2:
            return "Average"
        elif sharpe < 3:
            return "Good"
        else:
            return "Excellent"
    
    sharpe_interpretation = interpret_sharpe(annual_sharpe)

    return {
        'Total Return': total_return,
        'Win Rate': win_rate,
        'Average Profit': avg_profit,
        'Max Drawdown': max_drawdown,
        'Max Drawup': max_drawup,
        'Raw Sharpe Ratio': raw_sharpe,
        'Annualised Sharpe Ratio': annual_sharpe,
        'Sharpe Interpretation': sharpe_interpretation
    }

def display_summary_metrics(metrics, title="Simulation Metrics"):
    df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
    print(f"\n{title}:")
    print(df.to_string(index=False))
    # Optionally, print an interpretation for the Sharpe ratio.
    if 'Annualised Sharpe Ratio' in metrics and 'Sharpe Interpretation' in metrics:
        print(f"\nAnnualised Sharpe Ratio is {metrics['Annualised Sharpe Ratio']:.2f} which is considered {metrics['Sharpe Interpretation']}.")
