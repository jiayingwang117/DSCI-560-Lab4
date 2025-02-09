import torch.optim as optim
import torch.nn as nn
from lstm_model import LSTMModel
from config import INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, EPOCHS, LEARNING_RATE

# Train the LSTM Model
def train_lstm_model(train_loader):
    model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)
    criterion = nn.BCELoss()  # Binary classification loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item()}")

    return model