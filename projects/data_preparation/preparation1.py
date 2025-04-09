import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ------------------------
# Step 1: Load temperature data
# ------------------------
# Sample synthetic data (or use real one)
date_range = pd.date_range(start="2020-01-01", end="2022-12-31", freq="D")
temps = np.sin(np.linspace(0, 30 * np.pi, len(date_range))) * 10 + 20  # fake temp
df = pd.DataFrame({"Date": date_range, "Temp": temps})
df.set_index("Date", inplace=True)

# ------------------------
# Step 2: Normalize
# ------------------------
scaler = MinMaxScaler()
df['Temp_scaled'] = scaler.fit_transform(df[['Temp']])


# ------------------------
# Step 3: Create sequences
# ------------------------
SEQ_LEN = 30  # 30-day sequences

def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i:i+seq_len]
        y = data[i+seq_len]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X, y = create_sequences(df['Temp_scaled'].values, SEQ_LEN)
print("Shape of X:", X.shape)  # [samples, sequence_length]
print("Shape of y:", y.shape)  # [samples]

# ------------------------
# Step 4: Custom Dataset
# ------------------------
class TempDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # add feature dim
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = TempDataset(X, y)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Visual check on one sample
for batch_x, batch_y in loader:
    print("Batch X shape:", batch_x.shape)  # [batch_size, seq_len, input_size]
    print("Batch Y shape:", batch_y.shape)  # [batch_size]
    break
