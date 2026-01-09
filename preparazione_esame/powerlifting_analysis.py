# BestSquatKg → prevedere il miglior squat

import os
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers 
from tensorflow.keras import optimizers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


file_path = "../data/openpowerlifting.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError("Errore file CSV non trovato")

df = pd.read_csv(file_path, na_values=["NA", "-", "N/A", "NaN", ""], nrows=1000000)

print(df.info())
print(df.isnull().mean() * 100)

# Elimino subito la colonna MeetID, Nome dal momento che l'id di 
# un utente non può essere usato all'interno di un modello
df = df.drop(columns=["MeetID", "Name"])

# La colonna Squat4Kg, Deadlift4Kg e Bench4Kg
# sono per il 99% null quindi elimino la colonna
df = df.drop(columns=["Squat4Kg", "Bench4Kg", "Deadlift4Kg"])

la_sex = LabelEncoder()
df["Sex"] = la_sex.fit_transform(df["Sex"])

la_equipment = LabelEncoder()
df["Equipment"] = la_equipment.fit_transform(df["Equipment"])

la_division = LabelEncoder()
df["Division"] = la_division.fit_transform(df["Division"])

la_weight_class_kg = LabelEncoder()
df["WeightClassKg"] = la_weight_class_kg.fit_transform(df["WeightClassKg"])

la_place = LabelEncoder()
df["Place"] = la_place.fit_transform(df["Place"])

# Elimino tutte le righe vuote e anche tutte le righe in cui è 
# presente anche e solo un valore null
df_test = df.copy()
print(f"Dimensioni dataset PRIMA del dropna: {df.shape}")
df_test = df_test.dropna(axis=0, how="all")
df_test = df_test.dropna(axis=0, how="any")
print(f"Dimensioni dataset DOPO il dropna: {df_test.shape}")

# Dal momento che ho ridotto le righe del dataset di di 3 volte 
# controllo il grado di correlazione tra età e BestSquatKg se è
# basso al punto da poterlo escludere torno indietro e rimuovo 
# anche age prima di effettuare la rimozione delle righe NULL
num_cols = df_test.select_dtypes(include=[np.number]).columns.tolist()
correlation_matrix = df_test[num_cols].corr()

plt.figure(figsize=(12, 12))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=1.5)
plt.title("Correlation Matrix con AGE: è rilevante per il target?")
plt.show()

# Age è correlata per 0.03 con il bestSquatKg quindi direi che 
# posso rimuoverla e ridroppare le righe vuote o con dei vuoti
df = df.drop(columns=["Age"])
df = df.dropna(axis=0, how="all")
df = df.dropna(axis=0, how="any")
print(f"Dimensioni del dataset: {df.shape}")

# Così facendo ho ancora la metà dei dati di conseguenza 
# rifaccio la matrice di correlazione 
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
correlation_matrix = df[num_cols].corr()

plt.figure(figsize=(12, 12))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=1.5)
plt.title("Correlation Matrix con AGE: è rilevante per il target?")
plt.show()

# Le colonne ininfluenti sono Equipment(0.13), Division(0.04), Place(-0.05)
df = df.drop(columns=["Equipment", "Division",  "Place"])
num_cols.remove("Equipment")
num_cols.remove("Division")
num_cols.remove("Place")

# Scelgo di risolvere il problema con le reti neurali
# Dal momneto che ho un dataset con 10 feature circa
# Avrò 3 hidden layer da: 32 -> 16 -> 8 
target = "BestSquatKg"
X = df[num_cols]
y = df[target]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1], )),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(1, activation="linear")
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001), 
    loss="mse", 
    metrics=["mse"]
)

history = model.fit(
    X_train, y_train, 
    batch_size=128,
    epochs=50,
    verbose=1,
    validation_data=(X_val, y_val)
)

loss, mse = model.evaluate(X_test, y_test, verbose=0)
print(f"MSE del modello: {mse}")
y_pred = model.predict(X_test, verbose=0).flatten()

plt.figure(figsize=(12, 12))
plt.scatter(y_test, y_pred, color="blue", alpha=0.6, label="Predict")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", label="Linear Regression")
plt.title("Valutazione predict")
plt.legend()
plt.show()
