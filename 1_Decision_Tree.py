import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


# =================== Caricamento Dataset ===================
file_path = "data/air_quality.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError("Il file CSV non esisteva")

df = pd.read_csv(file_path, low_memory=False, na_values=['-', 'NA', 'ND', 'n/a', ''])


# ================== Dataset exploration =====================
print(f"Grandezza dataset originale {df.shape}")

print("\nPrime 5 righe del dataset")
print(df.head())

print("\nInformazioni del dataset")
print(df.info())

print("\nDescrizione statistiche")
print(df.describe(include="all"))

print("\nDistribuzione numerica delle colonne")
df.hist(figsize=(12, 6))
plt.tight_layout()
plt.show()


# ================== Data cleaning =====================
df = df.dropna(axis=1, how='all') # Eliminazione colonne vuote

print("\nPercentuale dei valori mancanti per colonna (dopo aver droppato le colonne vuote)")
print(df.isnull().mean() * 100)
print(f"Grandezza del dataset: {df.shape}")

df.dropna(subset='status', inplace=True)
print(df.isnull().mean() * 100)
print(f"\nDataset finale dopo aver rimosso le righe dove lo status non ha valore: {df.shape}")


# ================ Feature Transformation =================
# Semantic Mapping
print(df["status"].unique())

df["status"] = df["status"].replace({
    "Hazardous": 5,
    "Very Unhealthy": 4,
    "Unhealthy": 3,
    "Unhealthy for Sensitive Groups": 2,
    "Moderate": 1,
    "Good": 0
})

# Label Encoder
df["pollutant"] = df["pollutant"].fillna("Unknown")

le_pollutant = LabelEncoder()
df["pollutant"] = le_pollutant.fit_transform(df["pollutant"].astype(str))

le_sitename = LabelEncoder()
df["sitename"] = le_sitename.fit_transform(df["sitename"].astype(str))

le_county = LabelEncoder()
df["county"] = le_county.fit_transform(df["county"].astype(str))

# Discretization
bins = [0, 50, 100, 150, 200, 300, 500] # Intervalli di valori per rientrare nelle varie categorie
labels_num = [0, 1, 2, 3, 4, 5]  # 0=Good, ..., 5=Hazardous

df['aqi_discretized'] = pd.cut(
    df['aqi'],
    bins=bins,
    labels=labels_num,
    right=True,
    include_lowest=True
).astype('Int64')

df = df[df['aqi_discretized'].notna()]

