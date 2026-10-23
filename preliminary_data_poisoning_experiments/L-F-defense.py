from sklearn.ensemble import IsolationForest
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Generate some sample data
np.random.seed(0)
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)

# Introduce Data Poisoning Attack - Add outlier points to the dataset
poisoned_x = np.array([[0.5], [1.5], [2.5], [3.5], [4.5], [1.75], [0.2], [0.3], [0.7], [1.2], [1.2], [1.3]])
poisoned_y = np.array([[0.5], [1.5], [2.5], [0], [0.5], [1], [0.2], [0.3], [0.7], [1.2], [1.2], [0.3]])

# Append poisoned data to the original data
x = np.vstack([x, poisoned_x])
y = np.vstack([y, poisoned_y])

# Combine x and y for outlier detection
data = np.hstack([x, y])

# Use Isolation Forest to detect outliers
iso_forest = IsolationForest(contamination=0.1, random_state=42)
outliers = iso_forest.fit_predict(data)

# Filter out the detected outliers
x_filtered = x[outliers == 1]
y_filtered = y[outliers == 1]

# Split the filtered data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_filtered, y_filtered, test_size=0.2, random_state=42)

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
plt.scatter(x_filtered, y_filtered, color="green", label="Filtered data points (outliers removed)")
plt.plot(x_test, y_pred, color="red", linewidth=2, label="Regression line (after outlier removal)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
