# Link consegna: https://chatgpt.com/share/6967b589-33a8-8013-9629-1b61a852407b
#
# Attributi
# - V1: Recency → mesi dall’ultima donazione
# - V2: Frequency → numero totale di donazioni
# - V3: Monetary → quantità totale di sangue donato (in cc)
# - V4: Time → mesi dalla prima donazione
# Target:
# Variabile binaria che indica se il donatore ha donato sangue a marzo 2007
# - 2 = ha donato
# - 1 = non ha donato


import os 
import seaborn as sns
import numpy as np
import pandas as pd

import tensorflow.keras as keras
from tensorflow.keras import layers, optimizers

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, root_mean_squared_error, silhouette_score
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


path_file = "../data/blood.csv"
if not os.path.exists(path_file):
    raise FileNotFoundError("CSV not found file")
df = pd.read_csv(path_file, na_values=["NA", "", "-", "N/A", "NaN"], nrows=1000000)

print(f"\nDescrizione dataset: {df.shape}")
print(df.head())
print(df.info())
print(df.describe(include="all"))
print(df.isnull().mean() * 100)

print("\nDistribuzione numerica delle colonne")
df.hist()
plt.tight_layout()
plt.show()

# Sono tutte colonne numeriche: stampo la matrice di correlazione
cm = df.corr()
plt.figure(figsize=(12, 12))
sns.heatmap(cm, cmap="coolwarm", annot=True, fmt=".2f", linewidths=1.5)
plt.title("Matrice di correlazione")
plt.show()

target = "Class"
class_values = df[target].unique()
num_classes = len(class_values)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if target in num_cols:
    num_cols.remove(target)

# Rimuovo V4 perchè matrice di confusione ha come valore 0.04 quindi rimuovo il rumore
num_cols.remove("V4")     

# ========================= ALBERI DECISIONALI ==============================
X = df[num_cols]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print(f"Punteggio Decision Tree: {accuracy_score(y_test, y_pred)}")

plt.figure()
plot_tree(model, max_depth=3)
plt.show()

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.title("Confusion Matrix Class")
plt.show()


# ===================== PROVA CON RETE NEURALE =========================
y = y.replace({2: 1, 1: 0})

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)), 
    layers.Dense(16, activation="relu"), 
    layers.Dense(8, activation="relu"),
    layers.Dense(4, activation="relu"), 
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy", 
    metrics=["accuracy"]
)

history = model.fit(
    X_train_scaled, y_train, 
    batch_size=128, 
    epochs=50, 
    verbose=1,
    validation_data=(X_val_scaled, y_val)
)

loss, acc = model.evaluate(X_val_scaled, y_val, verbose=1)
print("\nMetriche Neural Networks:")
print(f"Loss: {loss}")
print(f"Accuracy: {acc}")

y_pred = (model.predict(X_test_scaled) > 0.5).astype(int).flatten()

plt.figure(figsize=(12, 12))
plt.plot(history.history["accuracy"], color="red", label="Accuracy")
plt.plot(history.history["val_accuracy"], color="blue", label="val_accuracy")
plt.title("Neural Network: accuracy vs val_accuracy")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.legend()
plt.show()

plt.figure(figsize=(12, 12))
plt.plot(history.history["loss"], color="red", label="loss")
plt.plot(history.history["val_loss"], color="blue", label="val_loss")
plt.title("Neural Network: loss vs val_loss")
plt.xlabel("Epochs")
plt.ylabel("Values")
plt.legend()
plt.show()

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.title("Confusion Matrix: Neural Network")
plt.show()


# ======================= Trasformo il problema in regressione lineare ============================
# Il nuovo target è V2 (Frequenza)

target = "V2"

cm = df.corr()
sns.heatmap(cm, cmap="coolwarm", annot=True, fmt=".2f", linewidths=1.5)
plt.title("Matrice di correlazione: Focus su V2")
plt.show()

# Tutte le variabili sono sufficientemente collegate, sono 
# tutte numeriche quindi procedo con lo splitting dei dati
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if target in num_cols:
    num_cols.remove(target)
    
X = df[num_cols]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

print("\nValutazione Linear Regression")
print(f"mse: {mse}")
print(f"rmse: {rmse}")

plt.figure(figsize=(12,12))
plt.scatter(y_test, y_pred, alpha=0.6, color="blue", label="Predicted")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", label="linear regression")
plt.title("Regressione lineare grafico")
plt.xlabel("Real Value")
plt.ylabel("Predicted Value")
plt.legend()
plt.show()

# # =========================== Soluzione con rete neurale =====================
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

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
    X_train_scaled, y_train, 
    batch_size=128, 
    epochs=50,
    verbose=1,
    validation_data=(X_val_scaled, y_val)
)

loss, mse = model.evaluate(X_val_scaled, y_val)
print(f"\nEvaluate Neural Network:")
print(f"loss: {loss}")
print(f"mse: {mse}")

y_pred = model.predict(X_test_scaled).flatten()

plt.figure(figsize=(12, 12))
plt.plot(history.history["mse"], color="red", label="mse")
plt.plot(history.history["val_mse"], color="blue", label="val_mse")
plt.title("Validazione mse e val_mse")
plt.xlabel("Epoche")
plt.ylabel("Value")
plt.legend()
plt.show()

plt.figure(figsize=(12, 12))
plt.plot(history.history["loss"], color="red", label="loss")
plt.plot(history.history["val_loss"], color="blue", label="val_loss")
plt.title("Validazione loss e val_loss")
plt.xlabel("Epoche")
plt.ylabel("Value")
plt.legend()
plt.show()

plt.figure(figsize=(12, 12))
plt.scatter(y_test, y_pred, color="blue", alpha=0.6, label="predict")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", label="linear regression")
plt.legend()
plt.show()



# ==================================== CLUSTERING ========================================
# Usa solo le feature V1, V2, V3, V4.
# Standardizza i dati
# - Applica K-Means con k da 2 a 6
# - Usa Elbow Method o Silhouette

X = df.drop(columns=["Class"])

scaler = StandardScaler()
X = scaler.fit_transform(X)

labels = [[] for i in range(5)]
scores = [None for i in range(5)]
num_clusters = 0
max_score = 0
for i in range(5):
    kmeans = KMeans(n_clusters=i+2, random_state=42, n_init=10)
    labels[i] = kmeans.fit_predict(X)
    scores[i] = silhouette_score(X, labels[i])
    if max_score < scores[i]: 
        max_score = scores[i]
        selected_label = labels[i]
        num_cluster = i + 2
    print(f"KMeans k={i+2}: {scores[i]}")
    
print(f"Selected Labels {selected_label}")

pca = PCA(n_components=2)
reduced = pca.fit_transform(X)
base_cmap = plt.colormaps["tab10"]
colors = base_cmap.colors[:num_clusters]
cmap = ListedColormap(colors)

plt.figure(figsize=(8, 6))
sc1 = plt.scatter(reduced[:, 0], reduced[:, 1], c=selected_label, cmap=cmap, alpha=0.6, s=12)

unique_clusters = np.unique(selected_label)

colors = [cmap(i) for i in range(num_cluster)]
patches = [mpatches.Patch(color=colors[i], label=f"cluster_{cl}") 
           for i, cl in enumerate(unique_clusters)]
plt.legend(handles=patches, title="labels", frameon=True)
sizes = np.bincount(selected_label)
sizes_txt = ", ".join(f"{i}:{sizes[i]}" for i in range(len(sizes)))

plt.title(f"KMeans PCA 2D (k={num_clusters}) | size [{sizes_txt}]")
plt.tight_layout()
plt.show()
