import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# 1. Download EUR/USD data
# -----------------------------
data = yf.download("EURUSD=X", start="2015-01-01", end="2024-12-31")
data = data[['Close']].dropna()

# Plot data
plt.figure(figsize=(12, 4))
plt.plot(data['Close'])
plt.title('EUR/USD Close Price')
plt.ylabel('USD')
plt.xlabel('Date')
plt.grid(True)
plt.show()

# -----------------------------
# 2. Preprocess Data
# -----------------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['Close']])

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LEN = 30
X, y = create_sequences(scaled_data, SEQ_LEN)

# Train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert to tensors and add input dimension
X_train = torch.tensor(X_train, dtype=torch.float32)  # shape: (batch, seq_len, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# -----------------------------
# 3. Define LSTM Model
# -----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)  # output shape: (batch, seq_len, hidden)
        out = self.fc(out[:, -1, :])  # take last time step
        return out

model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# 4. Train the Model
# -----------------------------
EPOCHS = 30
for epoch in range(EPOCHS):
    model.train()
    output = model(X_train).squeeze()
    loss = criterion(output, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}')

# -----------------------------
# 5. Make Predictions
# -----------------------------
model.eval()
with torch.no_grad():
    train_pred = model(X_train).squeeze().numpy()
    test_pred = model(X_test).squeeze().numpy()

# Inverse transform to original scale
train_pred = scaler.inverse_transform(train_pred.reshape(-1, 1)).flatten()
test_pred = scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# -----------------------------
# 6. Plot the Results
# -----------------------------
plt.figure(figsize=(14, 5))
plt.plot(range(len(y_train_actual)), y_train_actual, label='Actual Train')
plt.plot(range(len(train_pred)), train_pred, label='Predicted Train')
plt.plot(range(len(y_train_actual), len(y_train_actual)+len(y_test_actual)), y_test_actual, label='Actual Test')
plt.plot(range(len(train_pred), len(train_pred)+len(test_pred)), test_pred, label='Predicted Test')
plt.title("EUR/USD Forecasting using LSTM")
plt.xlabel("Days")
plt.ylabel("Exchange Rate")
plt.legend()
plt.grid(True)
plt.show()
