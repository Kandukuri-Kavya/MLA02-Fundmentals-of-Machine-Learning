import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Dataset
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([2, 5, 10, 17, 26, 37])

# Linear Regression
linear = LinearRegression()
linear.fit(X, y)
y_linear = linear.predict(X)

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)
y_poly = model.predict(X_poly)

# Display results
print("Linear Regression")
print("Predictions:", y_linear)
print("Mean Squared Error:", mean_squared_error(y, y_linear))

print("\nPolynomial Regression")
print("Predictions:", y_poly)
print("Mean Squared Error:", mean_squared_error(y, y_poly))

# Plot graphs
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, y_linear, color="red", label="Linear Regression")
plt.plot(X, y_poly, color="green", label="Polynomial Regression")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show()
