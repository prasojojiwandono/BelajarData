# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# Step 2: Load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'
df = pd.read_csv(url, parse_dates=['Date'])

# Step 3: Basic exploration
print(df.head())
print(df.describe())

# Step 4: Plot the data
plt.figure(figsize=(12, 4))
plt.plot(df['Date'], df['Temp'])
plt.title('Daily Minimum Temperatures')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.show()



# Step 1: Normalize temperature
scaler = MinMaxScaler()
temps = scaler.fit_transform(df[['Temp']].values)

# Step 2: Create sequence dataset
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LENGTH = 7  # Using 7 days to predict the next day
X, y = create_sequences(temps, SEQ_LENGTH)

# Step 3: Split into training and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Step 4: Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))  # LSTM output
        out = out[:, -1, :]              # Take the output of the last time step
        out = self.linear(out)           # Linear layer to map to output
        return out


model = LSTMModel()


# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Reshape inputs to (batch, seq_len, input_size)
X_train_seq = X_train.view(X_train.size(0), SEQ_LENGTH, 1)
X_test_seq = X_test.view(X_test.size(0), SEQ_LENGTH, 1)

# Training loop
EPOCHS = 50
for epoch in range(EPOCHS):
    model.train()
    outputs = model(X_train_seq)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    predictions = model(X_test_seq)

# Undo the normalization
predicted = scaler.inverse_transform(predictions.numpy())
actual = scaler.inverse_transform(y_test.numpy())

# Plot the results
plt.figure(figsize=(12, 4))
plt.plot(actual, label='Actual Temperatures')
plt.plot(predicted, label='Predicted Temperatures')
plt.legend()
plt.title('Temperature Prediction')
plt.xlabel('Days')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.show()
