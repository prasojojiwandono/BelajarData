import numpy as np
import matplotlib.pyplot as plt

# Simulated time series data (e.g., temperature)
np.random.seed(42)
data = np.sin(np.linspace(0, 20, 100)) + np.random.normal(0, 0.2, 100)  # 100 time steps

# print(data)

plt.figure(figsize=(10, 4))
plt.plot(data, label="Temperature")
plt.title("Raw Time Series Data")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

SEQ_LEN = 10

X = []
y = []

for i in range(len(data) - SEQ_LEN):
    X.append(data[i:i + SEQ_LEN])
    y.append(data[i + SEQ_LEN])

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)  # (90, 10)
print("y shape:", y.shape)  # (90,)

sample_idx = 5  # pick any sample

plt.figure(figsize=(8, 4))
plt.plot(range(SEQ_LEN), X[sample_idx], label="Input Sequence")
plt.scatter(SEQ_LEN, y[sample_idx], color='red', label="Target (Next Step)")
plt.title(f"Sample #{sample_idx} - Input vs Target")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

