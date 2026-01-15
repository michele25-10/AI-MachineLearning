# Target: Species --> Classificatore
# Step 1 – Analisi esplorativa dei dati
# Step 2 - Albero Decisionale
# Step 3 - Rete Neurale


import os 
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

import tensorflow.keras as keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

file_path = "../data/Iris.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"{file_path} non trovato")
df = pd.read_csv(file_path, na_values=["", "-", "N/A", "NaN", "NA"], nrows=10000000)


print(f"\nDescrizione del dataset: \n{df.info()}")
print(f"\nStatistiche del dataset: \n{df.describe()}")
print(f"\nValori None del dataset: \n{df.isnull().mean() * 100}")

df.hist()
plt.tight_layout()
plt.show()

df = df.drop(columns=["Id"])

target="Species"
num_classes = df[target].nunique()
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if target in num_cols:
    num_cols.remove(target)
    
# ONE HOT ENCODING Non va usato sul target con soluzioni ad albero

# ====================== NEURAL NETWORK =====================
# Prima prendo le y le trasformo in train, val e test poi passo
# ad eseguire Il label encoder su tutte le y: poi con to_categorical
# passo a gestire il one hot encoding
X = df[num_cols]
y = df[target]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

la_y = LabelEncoder()
y_train_idx = la_y.fit_transform(y_train)
y_val_idx = la_y.transform(y_val)
y_test_idx = la_y.transform(y_test)

y_train_oh = to_categorical(y_train_idx, num_classes=num_classes)
y_test_oh = to_categorical(y_test_idx, num_classes=num_classes)
y_val_oh = to_categorical(y_val_idx, num_classes=num_classes)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

model = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1], )),
    layers.Dense(16, activation="relu"), 
    layers.Dense(8, activation="relu"), 
    layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy", 
    metrics=["accuracy"]
)

history = model.fit(
    X_train_scaled, y_train_oh,
    batch_size=32, 
    epochs=50, 
    verbose=1, 
    validation_data=(X_val_scaled, y_val_oh)
)

loss, acc = model.evaluate(X_val_scaled, y_val_oh)
print("Neural Networks:")
print(f"Loss: {loss}")
print(f"Accuracy: {acc}")

y_pred = model.predict(X_test_scaled).argmax(axis=1)

plt.figure()
plt.plot(history.history["accuracy"], color="blue", label="accuracy")
plt.plot(history.history["val_accuracy"], color="red", label="val_accuracy")
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history["loss"], color="blue", label="loss")
plt.plot(history.history["val_loss"], color="red", label="val_loss")
plt.legend()
plt.show()

cm = confusion_matrix(y_test_idx, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=la_y.classes_)
disp.plot()
plt.show()
