from data_loader import fetch_stock_data, compute_indicators, add_true_label, create_dataloader
from train import train_lstm_model
from trading_simulation import simulate_trading, plot_performance, calculate_metrics

def main():
    symbol = 'NVDA'
    df = fetch_stock_data(symbol)
    df = compute_indicators(df)
    df = add_true_label(df)

    # Create DataLoader & Train Model
    train_loader = create_dataloader(df)
    model = train_lstm_model(train_loader)

    # Predict on Test Data
    test_dataset = create_dataloader(df)
    X_test, _ = test_dataset.dataset[:]
    y_pred = (model(X_test).detach().numpy() > 0.5).astype(int)

    # Simulate Trading Strategy (Updated)
    df = simulate_trading(df, y_pred)

    # Print Trading Performance Metrics
    calculate_metrics(df)

    # Plot Performance
    plot_performance(df)

if __name__ == "__main__":
    main()