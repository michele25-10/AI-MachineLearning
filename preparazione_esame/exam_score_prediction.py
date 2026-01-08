# Informazioni sul dataset

# Questo dataset fornisce una rappresentazione ampia e realistica dei vari fattori che contribuiscono 
# alle prestazioni degli studenti agli esami. Contiene 20.000 record, ognuno dei quali descrive il 
# comportamento accademico di uno studente, le sue abitudini di studio, le routine di vita e le 
# condizioni d’esame. Queste variabili aiutano collettivamente a comprendere come diversi aspetti della
# vita quotidiana e dell’ambiente di apprendimento influenzino i risultati agli esami.
# 
# Il dataset include dettagli come: ore di studio giornaliere, percentuale di frequenza alle lezioni,
# durata e qualità del sonno, disponibilità di internet, metodo di studio, valutazione delle strutture
# dell’istituto e livello di difficoltà dell’esame. Questi fattori riflettono un ampio spettro di 
# influenze comunemente osservate nei contesti educativi. Il punteggio d’esame (0–100) è ottenuto tramite
# una formula ponderata che riproduce modelli realistici di rendimento scolastico.

# Trova un modello per predire i risultati di un test

import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers, models

from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, root_mean_squared_error

file_path = "../data/Exam_Score_Prediction.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError("CSV non trovato")

df = pd.read_csv(file_path, na_values=['-', 'NA', 'N/A', 'NaN'], nrows=10000000)

print("\nDataset information: ")
print(df.info())

print(df.head())

df = df.drop(columns=["student_id"])
df = df.dropna(axis=1, how="all")
df = df.dropna(axis=1, how="any")

df["internet_access"] = df["internet_access"].replace({
    'yes': 1, 
    'no': 0
})

la_gender = LabelEncoder()
df["gender"] = la_gender.fit_transform(df["gender"])

la_course = LabelEncoder()
df["course"] = la_course.fit_transform(df["course"])

la_sleep_quality = LabelEncoder()
df["sleep_quality"] = la_sleep_quality.fit_transform(df["sleep_quality"])

la_study_method = LabelEncoder()
df["study_method"] = la_study_method.fit_transform(df["study_method"])

la_facility_rating = LabelEncoder()
df["facility_rating"] = la_facility_rating.fit_transform(df["facility_rating"])

la_exam_difficulty = LabelEncoder()
df["exam_difficulty"] = la_exam_difficulty.fit_transform(df["exam_difficulty"])

target = "exam_score"
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if target in num_cols:
    num_cols.remove(target)

numeric_cols = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_cols.corr()

plt.figure(figsize=(8,12))
sns.heatmap(correlation_matrix, annot=True, linewidths=1.5, cmap='coolwarm', fmt=".2f")
plt.title("Matrice di correlazione")
plt.show()

# Rimuovo le colonne con correlazione inesistente:
num_cols.remove("age")
num_cols.remove("gender")
num_cols.remove("course")
num_cols.remove("internet_access")
num_cols.remove("exam_difficulty")

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# ============== SPLIT, MODEL, PREDICT ==================
X = df[num_cols].copy()
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = root_mean_squared_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"\nDati analisi LinearRegression:")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")


plt.scatter(y_test, y_pred, alpha=0.6, color="Blue", label="Predict values")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", label= "Perfect prediction")
plt.ylabel("Risultato esame predicted")
plt.xlabel("Risulato esame reale")
plt.title("Linear Regression")
plt.grid(alpha=0.3)
plt.legend()
plt.show()


# ======================== Regressione Polinomiale ===========================
poly = make_pipeline(
    PolynomialFeatures(degree=3, include_bias=False),
    LinearRegression(),
)

poly.fit(X_train, y_train)
y_poly_pred = poly.predict(X_test)

rmse = root_mean_squared_error(y_test, y_poly_pred)
mse = mean_squared_error(y_test, y_poly_pred)

print("\nProva con regressione polinomiale")
print(f"mse: {mse}")
print(f"rmse: {rmse}")


# ==================== reti neurali =====================
X = df[num_cols]
y = df[target]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("\nReti Neurali:")
print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")

# Già standardizzato in precedenza
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_val_scaled = scaler.transform(X_val)
# X_test_scaled = scaler.transform(X_test)
X_train_scaled = X_train
X_val_scaled = X_val
X_test_scaled = X_test

model = models.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1], )),
    layers.Dense(128, activation="relu"),
    layers.Dense(128, activation="relu"), 
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="linear")
])

model.compile(
    optimizer="adam",
    loss="mse", 
    metrics=["mse"]    
)

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=64,
    batch_size=128,
    verbose=1
)

loss, mse = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test MSE: {mse:.4f}")

y_pred = model.predict(X_test_scaled, verbose=0).ravel()


# ========================= VISUALIZATION RESULT ==========================
plt.scatter(y_test, y_pred, alpha=0.6, color="Violet", label="Predict values neural networks")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", label= "Perfect prediction")
plt.ylabel("Risultato esame predicted")
plt.xlabel("Risulato esame reale")
plt.title("Neural Networks")
plt.grid(alpha=0.3)
plt.legend()
plt.show()