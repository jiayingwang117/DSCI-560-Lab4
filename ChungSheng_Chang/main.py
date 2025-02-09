from data_loader import fetch_stock_data, compute_indicators, add_true_label, create_dataloader
from train import train_lstm_model
from trading_simulation import simulate_trading, plot_performance

def main():
    symbol = 'NVDA'
    df = fetch_stock_data(symbol)
    df = compute_indicators(df)
    df = add_true_label(df)

    train_loader = create_dataloader(df)

    model = train_lstm_model(train_loader)

    test_dataset = create_dataloader(df)
    X_test, _ = test_dataset.dataset[:]
    y_pred = (model(X_test).detach().numpy() > 0.5).astype(int)

    df = simulate_trading(df, y_pred)
    plot_performance(df)

if __name__ == "__main__":
    main()