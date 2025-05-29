import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.read_csv("Housing.csv")
print("First 5 rows of the dataset:\n", df.head())
df_encoded = pd.get_dummies(df, drop_first=True)
y = df_encoded['price']
X_simple = df_encoded[['area']]
X_train_simple, X_test_simple, y_train, y_test = train_test_split(X_simple, y, test_size=0.2, random_state=42)
model_simple = LinearRegression()
model_simple.fit(X_train_simple, y_train)
y_pred_simple = model_simple.predict(X_test_simple)
print("\nSimple Linear Regression Evaluation:")
print("MAE:", mean_absolute_error(y_test, y_pred_simple))
print("MSE:", mean_squared_error(y_test, y_pred_simple))
print("R² Score:", r2_score(y_test, y_pred_simple))
plt.scatter(X_test_simple, y_test, color='yellow', label='Actual')
plt.plot(X_test_simple, y_pred_simple, color='green', linewidth=2, label='Predicted')
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Simple Linear Regression: Area vs Price")
plt.legend()
plt.grid(True)
plt.show()
features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
X_multiple = df_encoded[features]
X_train_multiple, X_test_multiple, y_train, y_test = train_test_split(X_multiple, y, test_size=0.2, random_state=42)
model_multiple = LinearRegression()
model_multiple.fit(X_train_multiple, y_train)
y_pred_multiple = model_multiple.predict(X_test_multiple)
print("\nMultiple Linear Regression Evaluation:")
print("MAE:", mean_absolute_error(y_test, y_pred_multiple))
print("MSE:", mean_squared_error(y_test, y_pred_multiple))
print("R² Score:", r2_score(y_test, y_pred_multiple))
print("\nMultiple Linear Regression Coefficients:")
print("Intercept:", model_multiple.intercept_)
for feature, coef in zip(features, model_multiple.coef_):
    print(f"{feature}: {coef:.2f}")
plt.scatter(y_test, y_pred_multiple, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Multiple Linear Regression: Actual vs Predicted")
plt.grid(True)
plt.show()
