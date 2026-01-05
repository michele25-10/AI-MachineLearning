# In this notebook, we apply linear regression to predict the 
# Air Quality Index (AQI) considring the dataset 
# Taiwan Air Quality Index Data 2016~2024.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

df.drop(axis=1, how="all")

df.dropna(inplace=True, subset=["pollutant", "aqi"])