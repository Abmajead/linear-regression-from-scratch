import numpy as np
from model import LinearRegression
import matplotlib.pyplot as plt

# Example dataset: y = 2x + 3 + noise
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2 * X.flatten() + 3 + np.random.randn(100) * 1.5

# Initialize and train model
model = LinearRegression(lr=0.01, n_iters=1000)
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Plot actual vs predicted
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred, color='red', label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression From Scratch')
plt.legend()
plt.show()
