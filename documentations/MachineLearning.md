# Machine Learning

## Guida Passo Passo

### 1. Import del csv

```python
# Controllo se il file esista
file_path = "data/air_quality.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError("Il file CSV non esisteva")

df = pd.read_csv(file_path, low_memory=False, na_values=['-', 'NA', 'ND', 'n/a', ''], nrows=1000000)
```

### 2. Esplorazione del dataset:

- `df.shape`: restituisce la dimensione del DataFrame.
- `df.head()`: mostra le prime 5 righe del dataset (di default).
- `df.info()`: mostra informazioni strutturali del DataFrame (tipo di dato, nomi colonne, numero di valori null, numero di righe).
- `df.describe(include="all")`: calcola statistiche descrittive per tutte le colonne.
- `df.hist(figsize=(12, 6))`: disegna istogrammi per tutte le colonne numeriche
- `df.isnull().mean() * 100`: isnull() restituisce un dataframe delle stesse dimensioni iniziali ma con True dove c'è un valore mancante e con False dove c'è un valore mancante mentre invece .mean() calcola la media colonna per colonna.
- `df[["pollutant", "status", "county", "sitename"]].dtypes`: restituisce il tipo per ogni colonna presente nella lista
- `df["status"].value_counts()`: valori presenti all'interno dello status vengono contati con quante volte appaiono
- `df["status"].unique()`: restituisce una lista dei valori di una specifica colonna di un dataframe

### 3. Pulizia dei dati

> Colonne vuote quando preoccuparsi:
>
> - `0% – 5% → NON preoccupante` in genere si può ignorare o imputare facilmente
> - `5% – 15% → ATTENZIONE` serve una valutazione consapevole (la variabile è importante?)
> - `15% – 30% → PROBLEMATICA` non si può ignorare
> - `> 30% → CRITICA / PREOCCUPANTE`

Eliminazione dati NaN:

- `df = df.dropna(axis=1, how='all')`: elimina le colonne completamente vuote
- `df.dropna(subset=['status'], inplace=True)`: eliminare righe con NaN in una colonna specifica
- `df.dropna(inplace=True)`: eliminare righe con almeno un NaN
- `df.dropna(how='all', inplace=True)` eliminare righe con tutti NaN
- `df.dropna(subset=num_cols, how='any')` eliminare le righ con NaN come valore solo nelle colonne numeriche

Imputazione (sostituire valori null con stime):

- `df['pm10'].fillna(0, inplace=True)`: sostituire valori NaN con un valore fisso
- `df['pm10'].fillna(df['pm10'].median(), inplace=True)`: sostituire valori NaN con la mediana
- `df['status'].fillna(df['status'].mode()[0], inplace=True)`: sostituire valori NaN con la moda

Rimozione dati duplicati:

- `df.drop_duplicates(inplace=True)`: eliminare righe duplicate
- `df.drop_duplicates(subset=['date', 'siteid'], inplace=True)`: eliminare duplicati basandosi su colonne specifiche

### 4. Feature Transformations

| Tipo di variabile | Modello sensibile all’ordine? | Encoding consigliato            |
| ----------------- | ----------------------------- | ------------------------------- |
| Ordinale          | Sì                            | LabelEncoder o mapping numerico |
| Nominale          | No                            | One-Hot Encoding                |
| Nominale ma ID    | No                            | LabelEncoder                    |

#### One-Hot encoding

```python
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer

# Applico OneHotEncoder su "status"
encoder = OneHotEncoder(sparse_output=False, drop=None)  # drop=None = mantengo tutte le colonne
status_reshaped = df_alt['status'].values.reshape(-1, 1)
status_onehot = encoder.fit_transform(status_reshaped)

# Creo un dataframe che contiene l'on-hot encoded status
status_onehot_df = pd.DataFrame(
    status_onehot,
    columns=[f"status_{cat}" for cat in encoder.categories_[0]],
    index=df_alt.index
)

# Aggiungo il risultato a df_alt
df_alt = pd.concat([df_alt, status_onehot_df], axis=1)
```

#### Semantic Mapping

Utile specialmente in contesti in cui è importante anche l'ordine logico (Hazardous > Very Unhealthy > Unhealty > .... > Good)

```python
# Visualizzo tutti i possibili valori di questa colonna
print(df["status"].unique())

# Qui vado a mappare e sostituire tutti i valori di questa colonna con un valore numerico
df["status"] = df["status"].replace({
    "Hazardous": 5,
    "Very Unhealthy": 4,
    "Unhealthy": 3,
    "Unhealthy for Sensitive Groups": 2,
    "Moderate": 1,
    "Good": 0
})

print(df)
```

#### Label Encoder

Trasformare variabili categoriche (anche ordinali) testuali (pollutant, county, sitename) in numeri interi per poterle usare nei modelli di machine learning.

```python
from sklearn.preprocessing import LabelEncoder  # Automatic with LabelEncoder

# Sostituisco il valore mancante 'pollutant'
df["pollutant"] = df["pollutant"].fillna("Unknown")

le_pollutant = LabelEncoder()
df["pollutant"] = le_pollutant.fit_transform(df["pollutant"].astype(str))

le_county = LabelEncoder()
df["county"] = le_county.fit_transform(df["county"].astype(str))
```

#### Discretization

Trasformare una variabile continua in una variabile categorica o ordinata, suddividendo l’intervallo dei valori in “bin” o intervalli predefiniti.
Invece di lavorare con valori numerici continui come aqi = 87.5, li raggruppiamo in categorie significative come Moderate.

```python
df_alt = df.copy()

# Discretization of AQI (column 'aqi')
bins = [0, 50, 100, 150, 200, 300, 500] # Intervalli di valori per rientrare nelle varie categorie
labels_num = [0, 1, 2, 3, 4, 5]  # 0=Good, ..., 5=Hazardous

# Directly create the numeric column for AQI ranges
df_alt['aqi_discretized'] = pd.cut(
    df_alt['aqi'],
    bins=bins,
    labels=labels_num,
    right=True,
    include_lowest=True
).astype('Int64')

df_alt = df_alt[df_alt['aqi_discretized'].notna()]

```

#### Conversioni

> Regole d'oro:
>
> - Se una colonna rappresenta tempo, va sempre convertita in datetime; poi estrai solo ciò che serve al modello (i modelli ML non capiscono il datetime direttamente).
>   | Feature | Quando è utile |
>   | ------- | -------------------------- |
>   | hour | fenomeni giornalieri |
>   | weekday | differenze feriali/weekend |
>   | month | stagionalità |
>   | year | trend lunghi |
>
> * Tutto ciò che è quantità, misura, valore fisico → numerico

```python
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['hour'] = df['date'].dt.hour
df['no2'] = pd.to_numeric(df['no2'], errors='coerce')
df['o3'] = pd.to_numeric(df['o3'], errors='coerce')

```

### 5. Visualizzazione

La visualizzazione serve a:

- Capire la distribuzione dei dati
- Controllare pattern e relazioni tra variabili
- Individuare errori, valori anomali o classi sbilanciate
- Comunicare i risultati in modo chiaro

#### Quale grafico usare?

| Tipo di dato / Obiettivo                      | Grafico consigliato    | Perché                                |
| --------------------------------------------- | ---------------------- | ------------------------------------- |
| Due variabili numeriche (trend, errori)       | Scatter plot           | Mostra correlazioni o discrepanze     |
| Distribuzione di una variabile numerica       | Histogram / KDE        | Capire range e frequenze              |
| Relazione tra variabile categorica e numerica | Box plot / Violin plot | Visualizza mediana, quartili, outlier |
| Variabile categorica                          | Bar chart              | Frequenze di ogni classe              |
| Classificazione modello                       | Confusion matrix       | Mostra accuratezza per classe         |
| Correlazione tra molte variabili numeriche    | Heatmap                | Evidenzia correlazioni alte/basse     |

#### Quale algoritmo usare?

1. Tipo di target
   - Continuo → regressione (LinearRegression, DecisionTreeRegressor, RandomForestRegressor…)
   - Categorico → classificazione (DecisionTreeClassifier, RandomForestClassifier, KNN, SVM…)
2. Distribuzione dei dati
   - Se target ordinato → Tree-based o ordinal regression
   - Se target nominale → One-Hot + classificatori generici
3. Numero di campioni e caratteristiche
   - Dataset piccolo → modelli semplici (Decision Tree, Logistic Regression)
   - Dataset grande → modelli complessi (Random Forest, Gradient Boosting, XGBoost)
4. Presenza di dati categorici
   - Molti categorici → Tree-based funziona bene con LabelEncoder
   - Modelli basati su distanza → meglio One-Hot Encoding
5. Interpretabilità
   - Serve spiegare → Decision Tree, Logistic Regression
   - Serve massima accuratezza → Random Forest, Gradient Boosting, Neural Network

#### Scatter plot

Comandi per i grafici:

- `plt.figure(figsize=(12,6))`: definisce le dimensioni del grafico
- `plt.scatter(X, Y, color='')`: disegna punti X vs Y con colore
- `plt.xlabel()`: etichetta asse X
- `plt.ylabel()`: etichetta asse Y
- `plt.title()`: dare un titolo al grafico
- `plt.legend()`: mostra legenda dei colori
- `plt.show()`: mostra il grafico a video

> Regola pratica:
>
> - Scatter plot → due variabili numeriche, trend o errori
> - Usa colori diversi per evidenziare pattern (match/mismatch, classi, anomalie)

#### Matrice di confusione

Comandi per i grafici:

- `confusion_matrix(y_true, y_pred, normalize='true')`: calcola la matrice normalizzata per riga
- `ConfusionMatrixDisplay`: classe per visualizzare la matrice in modo leggibile
- `disp.plot(cmap='Blues')`: mostra la matrice con una scala di colori
- `plt.show()`: mostra grafico

> Regole pratiche:
>
> - Matrice di confusione → classificazione
> - Normalizzare per riga → più facile confrontare classi sbilanciate
> - Colori → evidenziano accuratezza e errori

#### Matrice di correlazione

Una correlation matrix è una tabella (spesso visualizzata come heatmap) che mostra: quanto due variabili numeriche sono linearmente correlate tra loro. Ogni cella contiene un valore [-1, 1]:

- +1 → crescono insieme
- -1 → una cresce, l’altra diminuisce
- 0 → nessuna relazione lineare

Guardando la colonna (o riga) del target (status, aqi_discretized, ecc.) puoi vedere:

- quali variabili influenzano di più il target
- quali hanno poco o nessun legame

> Regola d'oro:
>
> - |r| ≥ 0.70: correlazione realmente forte (altamente predidttiva)
> - 0.40 ≤ |r| < 0.70: correlazione moderata (feature informativa) spesso migliora le performance, può lavorare bene con altre
> - 0.20 ≤ |r| < 0.40: correlazione debole ma utile per: modelli non lineari, modelli ad albero, interazioni tra feature
> - |r| < 0.20: nessun valore informativo (rumore, feature irrilevante)

```python

numeric_df = df.select_dtypes(include=['float64', 'int64'])
import seaborn as sns

correlation_matrix = numeric_df.corr()

# Plot heatmap
plt.figure(figsize=(14, 10))  # Define 14×10 inch figure for clear visualization
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap: Environmental Variables vs Status")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()  # Optimize spacing and margins
plt.show()

```

### 6. Rimuovi le features con bassa correlazione

Il **target leakage** avviene quando: una feature contiene informazioni che non sarebbero disponibili al momento della previsione (il modello impara una scorciatoia illegittima); Chiediti SEMPRE: Questa feature sarebbe disponibile prima di dover fare la previsione?.
Oppure segui le regole d'oro della matrice della correlazione per capire quali feature eliminare.

### 7. Split and Scale

```python
X = df_group[features]
y = df_group['status']

# Split dataset
X_train_group, X_test_group, y_train_group, y_test_group = train_test_split(
    X, y,
    test_size=0.3,   # 30% for test set and 70% for training set
    random_state=42  # for reproducibility
)

# Scaling
scaler = StandardScaler()
X_train_scaled_group = scaler.fit_transform(X_train_group)
X_test_scaled_group  = scaler.transform(X_test_group)

print("\nTraining set shape:", X_train_group.shape)
print("Testing set shape: ", X_test_group.shape)
print("Shapes after scaling:", X_train_scaled_group.shape, X_test_scaled_group.shape)

```

#### Standard Scaler

StandardScaler è uno strumento di feature scaling che trasforma ogni feature in modo che:

- media = 0
- deviazione standard = 1

x_scaled = (x - media)/deviazione standard

Questa trasformazione si chiama standardizzazione.

#### Algoritmi che NECESSITANO di StandardScaler

| Algoritmo           | Perché           |
| ------------------- | ---------------- |
| Logistic Regression | Gradient descent |
| SVM                 | Margini          |
| k-NN                | Distanza         |
| PCA                 | Varianza         |
| Neural Networks     | Stabilità        |

#### Algoritmi che NON lo richiedono

| Algoritmo     | Motivo                |
| ------------- | --------------------- |
| Decision Tree | Split su soglie       |
| Random Forest | Alberi indipendenti   |
| XGBoost       | Invarianti alla scala |
