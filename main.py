import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load Dataset
df = pd.read_csv("dataset/CarPrice_Assignment.csv")

print("Dataset Shape:")
print(df.shape)

# Remove ID column
df = df.drop("car_ID", axis=1)

# Encode categorical columns
label_encoder = LabelEncoder()

categorical_columns = df.select_dtypes(
    include=["object"]
).columns

for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Features and Target
X = df.drop("price", axis=1)
y = df["price"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("\nTraining model...")

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nModel Evaluation")

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Actual vs Predicted Plot
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")

plt.show()

# Feature Importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# Top 10 Features Graph
plt.figure(figsize=(10, 6))

plt.bar(
    feature_importance["Feature"][:10],
    feature_importance["Importance"][:10]
)

plt.xticks(rotation=45)

plt.title("Top 10 Features Affecting Car Price")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.tight_layout()

plt.show()