import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# =================== Caricamento Dataset ===================
file_path = "data/air_quality.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError("Il file CSV non esisteva")

df = pd.read_csv(file_path, low_memory=False, na_values=['-', 'NA', 'ND', 'n/a', ''], nrows=1000000)


# ================== Dataset exploration =====================
print(f"Grandezza dataset originale {df.shape}")

print("\nPrime 5 righe del dataset")
print(df.head())

print("\nInformazioni del dataset")
print(df.info())

print("\nDescrizione statistiche")
print(df.describe(include="all"))

#print("\nDistribuzione numerica delle colonne")
#df.hist(figsize=(12, 6))
#plt.tight_layout()
#plt.show()


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

# Conversion
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['hour'] = df['date'].dt.hour
df['no2'] = pd.to_numeric(df['no2'], errors='coerce')
df['o3'] = pd.to_numeric(df['o3'], errors='coerce')


# ======================= Visualizzazione dati =========================
numeric_df = df.select_dtypes(include=['float64', 'int64'])

correlation_matrix = numeric_df.corr()

# Plot heatmap
plt.figure(figsize=(14, 10))  # Define 14×10 inch figure for clear visualization
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap: Environmental Variables vs Status")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()  # Optimize spacing and margins
plt.show()


# ===================== Rimozione feature non influenti =======================
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['year'] = df['date'].dt.year

features = [
    'pm2.5', 'pm10',
    'co', 'co_8hr',
    'no2', 'nox', 'no',
    'so2',
    'o3',
    'windspeed', 'winddirec',
    'year', 'longitude', 'latitude'
    ]


# ======================= Split and Scale =========================
X = df[features]
y = df["status"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

print("\nTraining set shape:", X_train.shape)
print("Testing set shape: ", X_test.shape)
print("Shapes after scaling:", X_train_scaled.shape, X_test_scaled.shape)


# ====================== Model ========================
# Inizializzo il modello
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Predizioni
y_pred = model.predict(X_test_scaled)

# Valutazioni
print("\nModel Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Visualizzazione del Decision Tree
print("Features used:", X.columns.tolist())

plt.figure(figsize=(20, 10))
plot_tree(
    model,
    max_depth=2,
    feature_names=X.columns,
    class_names=[str(cls) for cls in sorted(y.unique())],
    filled=True,
    rounded=True,
    fontsize=5  
)
plt.title("Decision Tree (Depth ≤ 2)")
plt.show()

# Visualizzazione matrice di confusione
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(y.unique()))
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()