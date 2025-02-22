import os
import matplotlib.pyplot as plt

def plot_all_pairs(aggregated_data, filename="ensemble_model_simulation.png"):
    plt.figure(figsize=(12,8))
    symbols = aggregated_data['symbol'].unique()
    for sym in symbols:
        df_sym = aggregated_data[aggregated_data['symbol'] == sym]
        plt.plot(df_sym.index, df_sym['Close'], label=sym)
    plt.title("Aggregated Price Simulation (All Pairs)")
    plt.xlabel("Timestamp")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_all_pairs_with_trades(aggregated_data, trades_df, filename="ensemble_model_simulation.png"):
    plt.figure(figsize=(12,8))
    symbols = aggregated_data['symbol'].unique()
    for sym in symbols:
        df_sym = aggregated_data[aggregated_data['symbol'] == sym]
        plt.plot(df_sym.index, df_sym['Close'], label=sym)
    if not trades_df.empty:
        for _, trade in trades_df.iterrows():
            timestamp = trade['timestamp']
            price = trade['price']
            action = trade['action']
            if action.upper() == "BUY":
                plt.scatter(timestamp, price, marker="^", color="green", s=100, label="Buy")
            elif action.upper() == "SELL":
                plt.scatter(timestamp, price, marker="v", color="red", s=100, label="Sell")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.title("Aggregated Price Simulation with Trades (All Pairs)")
    plt.xlabel("Timestamp")
    plt.ylabel("Price")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_performance_for_each_pair(aggregated_data, trades_df, output_folder="plots"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    symbol_plots = {}
    symbols = aggregated_data['symbol'].unique()
    for sym in symbols:
        df_sym = aggregated_data[aggregated_data['symbol'] == sym]
        plt.figure(figsize=(12,8))
        plt.plot(df_sym.index, df_sym['Close'], label=f"{sym} Price")

        trades_sym = trades_df[trades_df['symbol'] == sym] if not trades_df.empty else None
        if trades_sym is not None and not trades_sym.empty:
            for _, trade in trades_sym.iterrows():
                timestamp = trade['timestamp']
                price = trade['price']
                action = trade['action']
                if action.upper() == "BUY":
                    plt.scatter(timestamp, price, marker="^", color="green", s=100, label="Buy")
                elif action.upper() == "SELL":
                    plt.scatter(timestamp, price, marker="v", color="red", s=100, label="Sell")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.title(f"{sym} Performance Diagram")
        plt.xlabel("Timestamp")
        plt.ylabel("Price")
        plt.grid(True)
        plt.tight_layout()
        filename = os.path.join(output_folder, f"performance_{sym}.png")
        plt.savefig(filename)
        plt.close()
        symbol_plots[sym] = filename
    return symbol_plots
