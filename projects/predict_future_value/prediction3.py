import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------
# 1. Load EUR/USD Data
# ---------------------------
data = yf.download("EURUSD=X", start="2015-01-01", end="2024-12-31")
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

# Create binary target
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
data = data.dropna()

# ---------------------------
# 2. Normalize Input Features
# ---------------------------
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data[feature_cols])
targets = data['Target'].values

# ---------------------------
# 3. Create Sequences
# ---------------------------
SEQ_LEN = 30

def create_sequences(data, targets, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = targets[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X, y = create_sequences(scaled_features, targets, SEQ_LEN)

# Train-test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# ---------------------------
# 4. Tuned LSTM Model
# ---------------------------
class TunedLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

model = TunedLSTM()
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ---------------------------
# 5. Train the Model
# ---------------------------
EPOCHS = 30
for epoch in range(EPOCHS):
    model.train()
    y_pred = model(X_train).squeeze()
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# ---------------------------
# 6. Evaluate Model
# ---------------------------
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test).squeeze().numpy()
    y_class = (y_pred_test >= 0.5).astype(int)

acc = accuracy_score(y_test, y_class)
cm = confusion_matrix(y_test, y_class)

print(f"\nTest Accuracy: {acc:.2%}")
print("Confusion Matrix:\n", cm)

# ---------------------------
# 7. Plot Confusion Matrix
# ---------------------------
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Predicted Down", "Predicted Up"],
            yticklabels=["Actual Down", "Actual Up"])
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.title('Confusion Matrix: Tuned LSTM')
plt.tight_layout()
plt.show()
