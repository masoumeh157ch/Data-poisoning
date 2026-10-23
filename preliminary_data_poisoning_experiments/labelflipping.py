# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate some sample data
np.random.seed(0)
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Introduce label flipping attack on a subset of the training data
flip_rate = 0.3  # 30% of labels will be flipped
num_flips = int(flip_rate * len(y_train))
flip_indices = np.random.choice(len(y_train), size=num_flips, replace=False)

# Flip the labels by negating the values (or you could add noise or other transformations)
y_train[flip_indices] = -y_train[flip_indices]

# Initialize the linear regression model
model = HuberRegressor()

# Train the model
model.fit(x_train, y_train.ravel())

# Make predictions
y_pred = model.predict(x_test)

print("True labels:", y_test)
print("Predicted labels:", y_pred)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Print the coefficients and intercept
slope = model.coef_[0]  # Extract the slope (m)
intercept = model.intercept_  # Extract the intercept (b)

print(f"Equation of the regression line: y = {slope:.2f}x + {intercept:.2f}")

# Plotting the results
plt.scatter(x, y, color="blue", label="Data points")
plt.plot(x_test, y_pred, color="red", linewidth=2, label="Regression line")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
