import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


"""
# Topics
    - ## Model creation and training process
    - ## Cost functions
    - ## Gradient descent
    - ## Maths behind gradient descent


# Machine Learning Training Steps (in general)
 
    Once we have the data ready!
    ---
    * Initialize weights
    * Repeat the following till the loss is minimized
        * Feed Data into the Model to predict
        * Calculate Loss
        * Update weights using Gradient Descent

"""

# Generate sample data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # 100 random values between 0 and 2
y = 3 * X.flatten() + 4 + np.random.randn(100)  # Linear relation with noise

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color="blue", alpha=0.6, edgecolors="k", label="Data points")
plt.title("Sample Data for Linear Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# Create DataFrame with X and y
df = pd.DataFrame({
    "X": X.flatten(),
    "y": y
})

# Display first 5 rows
print(df.head(5))

"""
MODEL: Y = W * X + B

### Show Linear Interactive
- With Bias
- Wihout Bias

"""

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_epochs=1000):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_epochs):

            # Predict
            y_predicted = np.dot(X, self.weights) + self.bias

            # Calculate gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Calculate loss
            loss = np.mean((y_predicted - y) ** 2)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Create and train model
model = LinearRegression(learning_rate=0.01, n_epochs=1000)
model.fit(X, y)

# Make predictions
predictions = model.predict([[5], [10], [15], [20]])

# Print results
print("Predictions:", predictions)
print("Learned weights:", model.weights)
print("Learned bias:", model.bias)

