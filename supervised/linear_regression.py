# Linear Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# Load the California housing dataset
print("Loading California Housing dataset...")
housing = fetch_california_housing()

# Create a DataFrame with the features
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

# Show dataset information
print(f"Dataset shape: {X.shape}")
print("\nFeature descriptions:")
for i, feature in enumerate(housing.feature_names):
    print(f"- {feature}: {housing.feature_names[i]}")

print("\nData preview:")
print(X.head())

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Create and train the linear regression model
print("\nTraining Linear Regression model...")
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel evaluation:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Display the coefficients
print("\nModel coefficients:")
for feature, coef in zip(housing.feature_names, model.coef_):
    print(f"- {feature}: {coef:.6f}")
print(f"- Intercept: {model.intercept_:.6f}")

# Visualize predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual prices')
plt.ylabel('Predicted prices')
plt.title('Linear Regression: Predicted vs Actual Housing Prices')
plt.tight_layout()
plt.savefig('linear_regression_results.png')
print("\nResults plot saved as 'linear_regression_results.png'")

if __name__ == "__main__":
    print("\nLinear regression analysis complete!") 