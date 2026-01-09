# CONSEGNA
# Classificazione su colonna quality_cat
# ===============================================

import os 
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import layers, optimizers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier


file_path = "../data/5b_winequality-white_cat.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError("CSV non trovato")

df = pd.read_csv(file_path, na_values=["NA", "-", "", "N/A", "NaN"], nrows=1000000)


print(df.info())
print(df.head())

# Vedo che tutte le colonne sono di tipo numerico fatta eccezione per
# la colonna quality_cat (target) della quale ci tengo a sapere quali 
# sono i valori che assume.
target = "quality_cat"
target_classes = df[target].unique()
target_classes = sorted(target_classes)
num_classes = len(target_classes)

print(f"\nValori assunti da quality_cat: {target_classes}")

la_target = LabelEncoder()
df[target] = la_target.fit_transform(df[target])

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

correlation_matrix = df[num_cols].corr()
plt.figure(figsize=(12, 24))
sns.heatmap(correlation_matrix, cmap="coolwarm", fmt=".2f", annot=True, linewidths=1.5)
plt.show()

if target in num_cols:
    num_cols.remove(target) 

# Rimuovo le variabili con bassa correlazione valori [-0.20, 0.20]
num_cols.remove("fixed_acidity")
num_cols.remove("residual_sugar")
num_cols.remove("citric_acid")
num_cols.remove("free_sulfur_dioxide")
num_cols.remove("pH")
num_cols.remove("sulphates")

X = df[num_cols]
y = df[target]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# 0.58
acc = accuracy_score(y_test, y_pred)

print(f"\nDecisionTreeClassifier: {acc:.2f}")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=sorted(y.unique()))
disp.plot(cmap="Blues")
plt.show()

# PROVO CON UN APPROCCIO DEEP LEARNING
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"), 
    layers.Dense(16, activation="relu"), 
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy", 
    metrics=["accuracy"] 
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=128,
    verbose=1 
)

loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Neural Network: {accuracy:.4f}")

y_pred = model.predict(X_test, verbose=0).argmax(axis=1) 

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=sorted(y.unique()))
disp.plot(cmap="Blues")
plt.show()


