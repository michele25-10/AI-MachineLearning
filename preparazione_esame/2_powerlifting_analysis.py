# Variabile target (classificatore): Equipment

import os 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

import tensorflow as tf
import tensorflow.keras as keras
import keras.layers as layers
import keras.optimizers as optimizers


file_path="../data/openpowerlifting.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError("CSV file non trovato")

df = pd.read_csv(file_path, na_values=["NA", "-", "NaN", "N/A", ""], nrows=1000000)

print("\nInformazioni Dataset:")
print(df.info())
print(df.head())

df = df.drop(columns=["MeetID"])
df = df.drop(columns=["Name"])

target = "Equipment"
classes = df[target].unique()
num_classes = len(classes)

print(f"Valori che può assumere il target: {classes}")

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

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cm = df[num_cols].corr()

plt.figure(figsize=(12,12))
sns.heatmap(cm, cmap="coolwarm", annot=True, fmt=".2f", linewidths=1.5)
plt.title("Correlation Matrix")
plt.show()

if target in num_cols:
    num_cols.remove(target)

# Rimuovo tutti gli elementi con una correlazione compresa tra (-20, 20)
num_cols.remove("Sex")
num_cols.remove("Age")
num_cols.remove("BodyweightKg")
num_cols.remove("WeightClassKg")
num_cols.remove("Squat4Kg"), 
num_cols.remove("BestSquatKg")
num_cols.remove("Bench4Kg")
num_cols.remove("BestBenchKg")
num_cols.remove("Deadlift4Kg")
num_cols.remove("BestDeadliftKg")
num_cols.remove("Place")

X = df[num_cols]
y = df[target]

X.dropna(axis=0, how="all")
X.dropna(axis=0, how="any")

scaler = StandardScaler()
X = scaler.fit_transform(X)

print(f"\nInformazioni dataset prima di addestrare il modello: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuratezza modello: {accuracy:.4f}")

cm = confusion_matrix(y_test, y_pred, )
disp = ConfusionMatrixDisplay(cm, display_labels=classes)
disp.plot()
plt.show()


# Procedo con la rete neurale per vedere il risultato migliore
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1], )),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(4, activation="relu"),
    layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train, y_train, 
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=128,
    verbose=1,
) 

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Loss: {test_loss}")
print(f"Accuracy: {test_acc}")

y_pred = model.predict(X_test).argmax(axis=1)

plt.figure(figsize=(12,12))
plt.plot(history.history.get("accuracy"), color="red", label="Accuracy")
plt.plot(history.history.get("val_accuracy"), color="blue", label="Val Accuracy")
plt.title("Accuracy Model")
plt.legend()
plt.show()

plt.figure(figsize=(12,12))
plt.plot(history.history.get("loss"), color="red", label="Loss")
plt.plot(history.history.get("val_loss"), color="blue", label="Val loss")
plt.title("Loss Model")
plt.legend()
plt.show()

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()

# Clustering
# kmeans = KMeans(n_clusters=num_classes, random_state=42)
# labels = kmeans.fit_predict(X)

# ct = pd.crosstab(df[target], labels, colnames=["cluster"])
# print("\nContingency: equipment vs cluster")
# print(ct)

# cluster_to_equipment = ct.idxmax(axis=0).to_dict()
