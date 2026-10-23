import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate some sample data
np.random.seed(0)
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)

# Function to apply gradient-based perturbations
def gradient_poisoning(x, y, model, learning_rate=0.1, num_iterations=10):
    poisoned_x = x.copy()
    poisoned_y = y.copy()

    for i in range(num_iterations):
        model.fit(poisoned_x, poisoned_y.ravel())  # Fit on poisoned data

        # Compute the gradient by calculating the residuals between predictions and true values
        predictions = model.predict(poisoned_x)
        residuals = predictions - poisoned_y.ravel()

        # Update poisoned_x and poisoned_y by following the gradient of the residuals
        poisoned_x += learning_rate * residuals.reshape(-1, 1) * np.sign(poisoned_x)
        poisoned_y += learning_rate * residuals.reshape(-1, 1) * np.sign(poisoned_y)

    return poisoned_x, poisoned_y

# Apply gradient-based poisoning
poisoned_x, poisoned_y = gradient_poisoning(x, y, HuberRegressor(), learning_rate=0.1, num_iterations=10)

# Combine original and poisoned data
x = np.vstack([x, poisoned_x])
y = np.vstack([y, poisoned_y])

# Split the poisoned data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize the Huber regression model
model = HuberRegressor()

# Train the model
model.fit(x_train, y_train.ravel())

# Make predictions
y_pred = model.predict(x_test)

# Display the true y_test and predicted y_pred values
print("True y_test values:\n", y_test)
print("============================================")
print("Predicted y_pred values:\n", y_pred)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Print the coefficients and intercept
slope = model.coef_[0]
intercept = model.intercept_

print(f"Equation of the regression line: y = {slope:.2f}x + {intercept:.2f}")

# Plotting the results
plt.scatter(x, y, color="blue", label="Data points including poisoned points")
plt.plot(x_test, y_pred, color="red", linewidth=2, label="Regression line (after poisoning)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
