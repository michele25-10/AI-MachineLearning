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
print(df.isnull().mean() * 100)

# Vedo che tutte le colonne sono di tipo numerico fatta eccezione per
# la colonna quality_cat (target) della quale ci tengo a sapere quali 
# sono i valori che assume.
target = "quality_cat"
df = df.dropna(subset=[target])
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = DecisionTreeClassifier()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# 0.58
acc = accuracy_score(y_test, y_pred)

print(f"\nDecisionTreeClassifier: {acc:.2f}")

labels=np.arange(len(la_target.classes_))
cm = confusion_matrix(y_test, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(cm, display_labels=la_target.classes_)
disp.plot(cmap="Blues")
plt.show()

# PROVO CON UN APPROCCIO DEEP LEARNING
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

model = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"), 
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy", 
    metrics=["accuracy"] 
)

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=75,
    batch_size=256,
    verbose=1 
)

loss, accuracy = model.evaluate(X_val_scaled, y_val, verbose=0)
print(f"Neural Network Accuracy: {accuracy:.4f}")

y_pred = model.predict(X_test_scaled, verbose=0).argmax(axis=1) 

# Grafici per valutazione del modello (Underfitting/Overfitting)
plt.figure(figsize=(12, 12))
plt.plot(history.history["accuracy"], color="red", label="Accuracy")
plt.plot(history.history["val_accuracy"], color="blue", label="Val Accuracy")
plt.title("Analisi Accuratezza")
plt.xlabel("Epoche")
plt.ylabel("Valore")
plt.legend()
plt.show()

plt.figure(figsize=(12, 12))
plt.plot(history.history["loss"], color="green", label="loss")
plt.plot(history.history["val_loss"], color="purple", label="Val loss")
plt.title("Analisi loss")
plt.xlabel("Epoche")
plt.ylabel("Valore")
plt.legend()
plt.show()

# Visualizzazione della matrice di confusione per poter giudicare output
labels = np.arange(len(la_target.classes_))
cm = confusion_matrix(y_test, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(cm, display_labels=la_target.classes_)
disp.plot(cmap="Blues")
plt.show()


