import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 1. Load the data
housing = fetch_california_housing(as_frame=True)
df = housing.frame

X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Fit an MLPRegressor
 # Increased from (64, 32) to (100, 50) to test whether a larger network improves fit
model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    early_stopping=True,
    max_iter=500,
    random_state=42
)
model.fit(X_train, y_train)

# 4. Predict on train and test
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

train_r2 = r2_score(y_train, train_preds)
test_r2 = r2_score(y_test, test_preds)

print("Train R2:", train_r2)
print("Test R2:", test_r2)

# 5. Plot: actual vs predicted (train)
plt.figure(figsize=(6, 6))
plt.scatter(y_train, train_preds, alpha=0.3, s=10)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
plt.xlabel("Actual (train)")
plt.ylabel("Predicted (train)")
plt.title("Train: Actual vs Predicted Median House Value")
plt.grid(alpha=0.3)
plt.text(
    0.05, 0.95, f"R$^2$ = {train_r2:.3f}",
    transform=plt.gca().transAxes,
    fontsize=11, verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)
plt.savefig("figures/train_actual_vs_pred.png", dpi=150, bbox_inches="tight")
plt.close()

# 6. Plot: actual vs predicted (test)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, test_preds, alpha=0.3, s=10, color="orange")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual (test)")
plt.ylabel("Predicted (test)")
plt.title("Test: Actual vs Predicted Median House Value")
plt.grid(alpha=0.3)
plt.text(
    0.05, 0.95, f"R$^2$ = {test_r2:.3f}",
    transform=plt.gca().transAxes,
    fontsize=11, verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)
plt.savefig("figures/test_actual_vs_pred.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved train_actual_vs_pred.png and test_actual_vs_pred.png")
