import torch
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

# Generate x values (100 samples between 0 and 1)
x = torch.rand(100, 1)

# Generate y values using y = 3x + 2 + noise
true_w = 3
true_b = 2
noise = 0.1 * torch.randn(100, 1)  # small random noise

y = true_w * x + true_b + noise

# Plot to visualize
plt.figure(figsize=(10, 4))
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated data')
plt.show()
