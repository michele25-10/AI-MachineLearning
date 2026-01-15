# Target: Species --> Classificatore
# Step 1 – Analisi esplorativa dei dati
# Step 2 - Albero Decisionale
# Step 3 - Rete Neurale
# Step 4 - Clustering
# Step 5 - Confronto 


import os
import pandas as pd
import numpy as np
import seaborn as sns

import tensorflow.keras as keras
from tensorflow.keras import layers, optimizers

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, silhouette_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA



file_path = "../data/Iris.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError("CSV file not found")
df = pd.read_csv(file_path, na_values=["", "NA", "NaN", "N/A", "-"], nrows=1000000)

print("\nDettagli dataset:")
print(df.info())
print(f"\nInformazioni statistiche:\n{df.describe()}")
print(f"\nAnalisi valori Null:\n{df.isnull().mean() * 100}")

df.hist()
plt.tight_layout()
plt.title("Analisi distribuzione dati del dataset")
plt.show()

# Da una prima analisi noto che non sono presenti colonne con dati NULL.
# Vedo però che la colonna di tipo ID non è rilevante,
# dal momento che è un identificatore della riga
df = df.drop(columns=["Id"])

# La variabile target Species è un object: Eseguo l'encode con LabelEncoder.
target = "Species"
classes = df[target].unique()
num_classes = len(classes)
la_species = LabelEncoder()
df[target] = la_species.fit_transform(df[target])

# Vedo che il dataset è perfettamente distribuito species ha 50 valori per ogni categoria
print(df[target].value_counts())

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cm = df[num_cols].corr()
sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm", linewidths=1.5)
plt.title("Matrice di correlazione")
plt.tight_layout()
plt.show()

# Dalla matrice di correlazione posso notare che tutte le 
# variabili sono correlate: tutti i valori sopra +-0.4

# Dal momento che non ho valori NULL non elimino righe
if target in num_cols:
    num_cols.remove(target)

# ==================== Decision Tree Classifier ======================
X = df[num_cols]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = DecisionTreeClassifier(max_depth=2, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print(f"\nDecisionTreeClassifier accuracy: {accuracy_score(y_test, y_pred)}")

plt.figure()
plot_tree(model, max_depth=2)
plt.show()

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=la_species.classes_)
disp.plot(cmap="plasma")
plt.title("Matrice confusione: DecisionTreeClassifier")
plt.show()

# Questo modello ha una accuratezza molto elevata 0.97 
# sempra essere perfetto considerato anche il max_depth basso


# ==================== Neural Networks ============================
X = df[num_cols]
y = df[target]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("\nControllo quali categorie sono presenti nei vari dataset:")
print(f"All value: {la_species.classes_}")
print(f"X_train value: {[la_species.classes_[i] for i in y_train.unique()]}")
print(f"X_val value: {[la_species.classes_[i] for i in y_val.unique()]}")
print(f"X_test value: {[la_species.classes_[i] for i in y_test.unique()]}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

model = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(16, activation="relu"), 
    layers.Dense(8, activation="relu"), 
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy", 
    metrics=["accuracy"]
)

history = model.fit(
    X_train_scaled, y_train, 
    batch_size=64, 
    epochs=75, 
    verbose=1, 
    validation_data=(X_val_scaled, y_val)
)

loss, acc = model.evaluate(X_val_scaled, y_val)
print("\nNeural Network:")
print("loss: ", loss)
print("accuracy: ", acc)

y_pred = model.predict(X_test_scaled).argmax(axis=1)

plt.figure()
plt.plot(history.history["accuracy"], color="blue", label="accuracy")
plt.plot(history.history["val_accuracy"], color="red", label="val_accuracy")
plt.title("accuracy vs val_accuracy")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history["loss"], color="blue", label="loss")
plt.plot(history.history["val_loss"], color="red", label="val_loss")
plt.title("loss vs val_loss")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.legend()
plt.show()

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=la_species.classes_)
disp.plot()
plt.show()

# Allora questo modello di neural network tende a unificare tutti
# i fiori sotto un unica classe (iris-virginica) così facendo non 
# è la soluzione migliore.
# Analizzando i grafici però dell'andamento delle val_loss e val_accuracy
# Posso dire con certezza che non c'è un overfitting del mio modello.
