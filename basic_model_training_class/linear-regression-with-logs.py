import numpy as np

# Generate sample data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # 100 random values between 0 and 2
y = 3 * X.flatten() + 4 + np.random.randn(100)  # Linear relation with noise

print("=" * 70)
print("STEP 1: DATA GENERATION")
print("=" * 70)
print(f"We generated {X.shape[0]} data points.")
print(f"X shape: {X.shape} (100 samples, 1 feature each)")
print(f"y shape: {y.shape} (100 target values)")
print(f"\nThe true relationship is: y = 3 * X + 4 + noise")
print(f"So ideally, the model should learn weight ≈ 3 and bias ≈ 4")
print(f"\nShowing first 5 data points:")
print(f"  {'X':>10}  {'y':>10}")
print(f"  {'-'*10}  {'-'*10}")
for i in range(5):
    print(f"  {X[i][0]:10.4f}  {y[i]:10.4f}")
print(f"  ... and {X.shape[0] - 5} more data points")
print(f"\nX range: [{X.min():.4f}, {X.max():.4f}]")
print(f"y range: [{y.min():.4f}, {y.max():.4f}]")

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_epochs=1000):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        print("\n" + "=" * 70)
        print("STEP 2: MODEL INITIALIZATION")
        print("=" * 70)
        print(f"Number of samples (n): {n_samples}")
        print(f"Number of features: {n_features}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Number of epochs: {self.n_epochs}")

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        print(f"\nInitial weights: {self.weights} (initialized to zeros)")
        print(f"Initial bias: {self.bias} (initialized to zero)")
        print(f"\nThis means the model starts as: y = 0 * X + 0 = 0")
        print(f"It will learn the correct weights and bias through training.")

        print("\n" + "=" * 70)
        print("STEP 3: TRAINING BEGINS (Gradient Descent)")
        print("=" * 70)
        print(f"The model will go through {self.n_epochs} epochs of training.")
        print(f"We will show details at epochs: 1, 2, 3, then every 100th epoch.\n")

        for epoch in range(self.n_epochs):
            show_log = epoch < 3 or (epoch + 1) % 100 == 0 or epoch == self.n_epochs - 1

            if show_log:
                print("-" * 70)
                print(f"EPOCH {epoch + 1}/{self.n_epochs}")
                print("-" * 70)
                print(f"  Current weights: {self.weights}")
                print(f"  Current bias: {self.bias:.6f}")

            # Predict
            y_predicted = np.dot(X, self.weights) + self.bias

            if show_log:
                print(f"\n  [Forward Pass] y_predicted = X * weights + bias")
                print(f"  Showing first 5 predictions vs actual:")
                print(f"    {'X':>10}  {'y_actual':>10}  {'y_predicted':>12}  {'error':>10}")
                print(f"    {'-'*10}  {'-'*10}  {'-'*12}  {'-'*10}")
                for i in range(5):
                    error = y_predicted[i] - y[i]
                    print(f"    {X[i][0]:10.4f}  {y[i]:10.4f}  {y_predicted[i]:12.4f}  {error:10.4f}")

            # Calculate gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            if show_log:
                print(f"\n  [Gradient Calculation]")
                print(f"  dw (gradient for weights) = (1/n) * X^T . (y_pred - y) = {dw}")
                print(f"  db (gradient for bias)     = (1/n) * sum(y_pred - y)   = {db:.6f}")

            # Update parameters
            old_weights = self.weights.copy()
            old_bias = self.bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if show_log:
                print(f"\n  [Parameter Update] using learning_rate = {self.learning_rate}")
                print(f"  weights: {old_weights} - {self.learning_rate} * {dw} = {self.weights}")
                print(f"  bias:    {old_bias:.6f} - {self.learning_rate} * {db:.6f} = {self.bias:.6f}")

            # Calculate loss
            loss = np.mean((y_predicted - y) ** 2)

            if show_log:
                print(f"\n  [Loss] MSE = mean((y_predicted - y)^2) = {loss:.6f}")
                print(f"  Updated weights: {self.weights}")
                print(f"  Updated bias: {self.bias:.6f}")
                print()

        print("=" * 70)
        print("STEP 4: TRAINING COMPLETE")
        print("=" * 70)
        print(f"Final learned weights: {self.weights}")
        print(f"Final learned bias: {self.bias:.6f}")
        print(f"Final loss (MSE): {loss:.6f}")
        print(f"\nThe model learned: y = {self.weights[0]:.4f} * X + {self.bias:.4f}")
        print(f"The true relation:  y = 3.0000 * X + 4.0000")
        print(f"Pretty close! The small difference is due to noise in the data.")

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Create and train model
print("\n" + "#" * 70)
print("LINEAR REGRESSION FROM SCRATCH - STEP BY STEP")
print("#" * 70)
print("\nWe are building a linear regression model from scratch.")
print("The model learns: y = weight * X + bias")
print("It uses Gradient Descent to find the best weight and bias.\n")

model = LinearRegression(learning_rate=0.01, n_epochs=1000)
model.fit(X, y)

# Make predictions
print("\n" + "=" * 70)
print("STEP 5: MAKING PREDICTIONS ON NEW DATA")
print("=" * 70)

new_data = [[5], [10], [15], [20]]
predictions = model.predict(new_data)

print(f"Using the learned model: y = {model.weights[0]:.4f} * X + {model.bias:.4f}")
print(f"\n  {'X_input':>10}  {'Predicted y':>12}  {'Expected (3X+4)':>16}")
print(f"  {'-'*10}  {'-'*12}  {'-'*16}")
for i, x_val in enumerate(new_data):
    expected = 3 * x_val[0] + 4
    print(f"  {x_val[0]:10d}  {predictions[i]:12.4f}  {expected:16.4f}")

print(f"\nFinal learned weights: {model.weights}")
print(f"Final learned bias: {model.bias:.6f}")
print("\n" + "#" * 70)
print("DONE! The model has successfully learned the linear relationship.")
print("#" * 70)
