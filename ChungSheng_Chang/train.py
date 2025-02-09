import torch
import torch.nn as nn
import torch.optim as optim
from lstm_model import LSTMModel
from config import INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, EPOCHS, LEARNING_RATE

# Train the LSTM Model
def train_lstm_model(train_loader):
    model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)
    criterion = nn.BCELoss()  # Binary classification loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()

            # Compute loss
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            # Convert predictions to binary (0 or 1)
            predicted = (y_pred > 0.5).float()
            
            # Compute accuracy
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
            total_loss += loss.item()

        # Compute average loss and accuracy
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    return model