# In this notebook, we apply linear regression to predict the 
# Air Quality Index (AQI) considring the dataset 
# Taiwan Air Quality Index Data 2016~2024.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split


# ================== Lettura dataframe =========================
file_path = "data/air_quality.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError("CSV non trovato nel file system")

df = pd.read_csv(file_path, na_values=['-', 'NA', 'ND', 'n/a', ''], nrows=1000000)

print(f"Informazioni del dataset {df.info()}")
print(f"Informazioni righe null {df.isnull().mean() * 100}")

df.dropna(axis=1, how="all", inplace=True)

le_county = LabelEncoder()
df["county"] = le_county.fit_transform(df["county"].astype(str))

le_sitename = LabelEncoder()
df["sitename"] = le_sitename.fit_transform(df["sitename"].astype(str))

le_pollutant = LabelEncoder()
df["pollutant"] = le_pollutant.fit_transform(df["pollutant"].astype(str))

# Seleziono tutte le colonne numeriche ed elimino le righe vuote
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df_clean = df.dropna(subset=numeric_cols).copy()
numeric_cols.remove('aqi')

X = df_clean[numeric_cols]
y = df_clean['aqi']

print("Le colonne numeriche selezionate:", numeric_cols)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

print("\nModel performance LINEAR REGRESSION:")
print(f"mse: {mse}")
print(f"rmse: {rmse}")

plt.scatter(y_test, y_pred, alpha=0.6, color="blue", label="Predicted values")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', label="Perfect prediction")
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Linear Regression: Actual AQI vs Predicted AQI")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ================= Polinomial Regression ====================
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

poly_model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression()
)

poly_model.fit(X_train, y_train)

y_poly_pred = poly_model.predict(X_test)

mse_poly = mean_squared_error(y_test, y_poly_pred)
rmse_poly = root_mean_squared_error(y_test, y_poly_pred)

print("\nModel performance POLY REGRESSION:")
print(f"mse: {mse_poly}")
print(f"rmse: {rmse_poly}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_poly_pred, alpha=0.6, color='blue', label="AQI Prediction")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', label="Perfect Match")
plt.title("Poly Regression degree=2: Predicted vs Actual")
plt.xlabel("Actual Value")
plt.ylabel("Predict Value")
plt.grid(alpha=0.3)
plt.legend()
plt.show()

# ==================== Ciclo: Find Best Model =========================
# Esercizio per capire quale modello è migliore

X_cols = [
    ['county', 'so2', 'co', 'o3', 'o3_8hr', 'pm10', 'pm2.5', 'no2', 'nox', 'no', 
        'windspeed', 'winddirec', 'co_8hr', 'pm2.5_avg', 'pm10_avg', 
        'so2_avg', 'longitude', 'latitude', 'siteid'
    ], 
    ['so2', 'co', 'o3', 'o3_8hr', 'pm10', 'pm2.5', 'no2', 'nox', 'no', 'windspeed', 
        'winddirec', 'co_8hr', 'pm2.5_avg', 'pm10_avg', 'so2_avg'
    ], 
    ['so2', 'co', 'o3', 'o3_8hr', 'pm10', 'pm2.5', 'no2', 'nox', 'no', 'windspeed', 'winddirec', 'co_8hr'], 
    ['so2', 'co', 'o3', 'o3_8hr', 'pm10', 'pm2.5', 'no2', 'nox', 'no', 'co_8hr']
]

models = [LinearRegression() for i in range(0,4)]
y_preds = [None for i in range(4)]
y_trains = []
y_tests = []
mses = [None for i in range(4)]
rmses = [None for i in range (4)]

for i in range(4):
    X_ex = X[X_cols[i]]
    
    X_train_ex, X_test_ex, y_train_ex, y_test_ex = train_test_split(X_ex, y, test_size=0.3, random_state=42)
    y_tests.append(y_test_ex)
    y_trains.append(y_train_ex)

    models[i].fit(X_train_ex, y_train_ex)
    y_preds[i] = models[i].predict(X_test_ex)
    
    rmses[i] = root_mean_squared_error(y_test_ex, y_preds[i])
    mses[i] = mean_squared_error(y_test_ex, y_preds[i])
    
    print(f"\nRisultati CICLO per migliorare il mio modello (iterazione {i}):")
    print(f"mse: {mses[i]}")
    print(f"rmse: {rmses[i]}")
    
color = ['blue', 'red', 'green', 'yellow']
for i in range(4):
    plt.scatter(y_tests[3-i], y_preds[3-i], alpha=0.6, color=color[i], label=f"Predict Linear Regression: test {i}")

plt.plot([y_tests[0].min(), y_tests[0].max()], [y_tests[0].min(), y_tests[0].max()], color='violet', label="Perfect prediction")
plt.title("Ciclo per controllare quale è il modello migliore!")
plt.xlabel("AQI value")
plt.ylabel("AQI predicted")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
