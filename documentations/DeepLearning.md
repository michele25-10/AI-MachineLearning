# Deep Learning

## Guida Passo Passo

### 1. Import delle librerie

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
```

### 2. Preparazione dei dati

#### Caricamento e preprocessing

```python
# Esempio con dataset personalizzato
X = df[features].values
y = df['target'].values

# Conversione target in formato categorico (per classificazione)
num_classes = len(np.unique(y))
y_categorical = to_categorical(y, num_classes=num_classes)

# Split train/validation/test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_categorical,
    test_size=0.3,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42
)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")
```

#### Normalizzazione

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

### 3. Costruzione del modello

#### Architettura Sequential

```python
model = models.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    # Hidden layers
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    # Output layer
    layers.Dense(num_classes, activation='softmax')  # softmax per classificazione
])
```

#### Numero di hidden layers

| Dataset            | Hidden layers |
| ------------------ | ------------- |
| Piccolo / semplice | 1–2           |
| Medio              | 2–3           |
| Complesso          | 3–5           |
| Deep learning vero | >5            |

> **Quanti neuroni per layer**?
> Il primo layer deve avere più neuroni delle feature per poter creare combinazioni non lineari utili:
> neuroni compresi tra i 2 x numero feature e 4 x numero feature

> **Regola d’oro:**
> Struttura piramidale decrescente; esempio:
> `128 → 64 → 32 → output`

#### Scelta delle activation functions

| Layer                                | Activation | Quando usarla               |
| ------------------------------------ | ---------- | --------------------------- |
| Hidden                               | ReLU       | Default per layer intermedi |
| Output (classificazione binaria)     | sigmoid    | Probabilità 0-1             |
| Output (classificazione multiclasse) | softmax    | Probabilità tra classi      |
| Output (regressione)                 | linear     | Valore continuo             |

### 4. Compilazione del modello

```python
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',  # per classificazione multiclasse
    metrics=['accuracy']
)
```

#### Scelta di optimizer e loss function

| Task                                         | Loss Function                     | Optimizer |
| -------------------------------------------- | --------------------------------- | --------- |
| Classificazione binaria                      | `binary_crossentropy`             | Adam      |
| Classificazione multiclasse One-Hot Encoding | `categorical_crossentropy`        | Adam      |
| Classificazione multiclasse Label Encoder    | `sparse_categorical_crossentropy` | Adam      |
| Regressione                                  | `mse`                             | Adam      |

#### Optimizer comuni

| Optimizer | Quando usarlo                                       |
| --------- | --------------------------------------------------- |
| Adam      | Default, funziona bene nella maggior parte dei casi |
| SGD       | Quando serve più controllo manuale                  |
| RMSprop   | Problemi con gradienti variabili                    |
| AdaGrad   | Dataset sparsi                                      |

### 5. Training del modello

```python
# Training
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=50,
    batch_size=128,
    verbose=1
)
```

#### Parametri importanti

| Parametro        | Significato                             | Valori tipici   |
| ---------------- | --------------------------------------- | --------------- |
| epochs           | Numero di passaggi completi sul dataset | 50-200          |
| batch_size       | Numero di campioni per aggiornamento    | 16, 32, 64, 128 |
| validation_split | % di dati per validazione               | 0.1-0.3         |

> Regole d'oro:
>
> - **Batch size piccolo** (16-32): più aggiornamenti, più rumore, training più lento
> - **Batch size grande** (128-256): meno aggiornamenti, convergenza più veloce, più memoria
> - **Early stopping**: previene overfitting fermando quando val_loss non migliora

### 6. Visualizzazione del training

```python
# Plot training history
plt.figure(figsize=(14, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

#### Interpretazione dei grafici

> La `loss` guida l'apprendimento (calcola quanto il modello ha sbagliato).
> La `metrics` servono per valutare le prestazioni in modo più intuitivo (calcola quanto il modello è bravo)

| Comportamento                      | Significato                   | Soluzione                                  |
| ---------------------------------- | ----------------------------- | ------------------------------------------ |
| Val_loss cresce, train_loss scende | **Overfitting**               | Più Dropout, Early stopping, Meno layer    |
| Entrambe alte e stabili            | **Underfitting**              | Più neuroni, Più layer, Training più lungo |
| Entrambe scendono insieme          | **Buon training**             | ✓ Continuare così                          |
| Val_loss oscilla molto             | **Batch size troppo piccolo** | Aumentare batch_size                       |

### 7. Valutazione del modello

```python
# Evaluation su test set
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Predictions
y_pred_prob = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)
```

#### Metriche di valutazione

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
```

### 8. Troubleshooting comuni

| Problema              | Causa probabile           | Soluzione                        |
| --------------------- | ------------------------- | -------------------------------- |
| Loss = NaN            | Learning rate troppo alta | Riduci LR a 0.0001               |
| Accuracy = 0 o random | Dati non normalizzati     | Applica StandardScaler           |
| Val_accuracy oscilla  | Batch size troppo piccolo | Aumenta a 64-128                 |
| Training molto lento  | Batch size troppo piccolo | Aumenta batch_size               |
| Memory error          | Batch size troppo grande  | Riduci batch_size                |
| Overfitting immediato | Troppi parametri          | Aggiungi Dropout, riduci neuroni |

---
