import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample data
np.random.seed(0)
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)

# Step 1: Outlier Detection and Filtering using IQR
def filter_outliers(x, y, threshold=1.5):
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    mask = (y >= lower_bound) & (y <= upper_bound)
    return x[mask[:, 0]], y[mask[:, 0]]

# Filter out outliers in y based on IQR
x_filtered, y_filtered = filter_outliers(x, y)

# Step 2: Gradient-based Poisoning with Gradient Clipping
def gradient_poisoning_with_clipping(x, y, model, learning_rate=0.1, num_iterations=10, clip_value=1.0):
    poisoned_x = x.copy()
    poisoned_y = y.copy()

    for i in range(num_iterations):
        model.fit(poisoned_x, poisoned_y.ravel())  # Fit on poisoned data

        # Calculate residuals between predictions and true values
        predictions = model.predict(poisoned_x)
        residuals = predictions - poisoned_y.ravel()

        # Apply Gradient Clipping
        clipped_residuals = np.clip(residuals, -clip_value, clip_value)

        # Update poisoned_x and poisoned_y using clipped gradients
        poisoned_x += learning_rate * clipped_residuals.reshape(-1, 1) * np.sign(poisoned_x)
        poisoned_y += learning_rate * clipped_residuals.reshape(-1, 1) * np.sign(poisoned_y)

    return poisoned_x, poisoned_y

# Apply poisoning with Gradient Clipping
poisoned_x, poisoned_y = gradient_poisoning_with_clipping(x_filtered, y_filtered, HuberRegressor(), learning_rate=0.1, num_iterations=10, clip_value=1.0)

# Combine original and poisoned data
x_combined = np.vstack([x_filtered, poisoned_x])
y_combined = np.vstack([y_filtered, poisoned_y])

# Split the poisoned data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_combined, y_combined, test_size=0.2, random_state=42)

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

slope = model.coef_[0]
intercept = model.intercept_

print(f"Equation of the regression line: y = {slope:.2f}x + {intercept:.2f}")

# Plotting the results
plt.scatter(x_combined, y_combined, color="blue", label="Data points including poisoned points")
plt.plot(x_test, y_pred, color="red", linewidth=2, label="Regression line (after defense)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
