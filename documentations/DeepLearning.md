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

#### Scelta delle activation functions

| Layer                                | Activation | Quando usarla                 |
| ------------------------------------ | ---------- | ----------------------------- |
| Hidden                               | ReLU       | Default per layer intermedi   |
| Hidden                               | LeakyReLU  | Se ReLU causa "dying neurons" |
| Hidden                               | tanh       | Dati centrati su 0            |
| Output (classificazione binaria)     | sigmoid    | Probabilità 0-1               |
| Output (classificazione multiclasse) | softmax    | Probabilità tra classi        |
| Output (regressione)                 | linear     | Valore continuo               |

#### Dropout

Il **Dropout** è una tecnica di regolarizzazione che:

- Disattiva casualmente una percentuale di neuroni durante il training
- Previene overfitting
- Forza la rete a imparare rappresentazioni più robuste

> Regole pratiche:
>
> - Dropout rate: 0.2-0.5 (20%-50%)
> - Non usare su layer di output
> - Maggiore dopo layer più grandi

### 4. Compilazione del modello

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # per classificazione multiclasse
    metrics=['accuracy']
)
```

#### Scelta di optimizer e loss function

| Task                        | Loss Function              | Optimizer |
| --------------------------- | -------------------------- | --------- |
| Classificazione binaria     | `binary_crossentropy`      | Adam      |
| Classificazione multiclasse | `categorical_crossentropy` | Adam      |
| Regressione                 | `mse` o `mae`              | Adam      |
| Regressione con outlier     | `huber`                    | Adam      |

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

### 8. Ottimizzazione del modello

#### Tecniche comuni

| Tecnica                      | Scopo                    | Quando usarla            |
| ---------------------------- | ------------------------ | ------------------------ |
| **Batch Normalization**      | Stabilizza training      | Reti profonde (5+ layer) |
| **Learning Rate Scheduling** | Riduce LR nel tempo      | Training molto lungo     |
| **Data Augmentation**        | Aumenta variabilità dati | Dataset piccoli          |
| **L1/L2 Regularization**     | Previene overfitting     | Molti parametri          |

#### Batch Normalization

```python
model = models.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),

    layers.Dense(128),
    layers.BatchNormalization(),  # Normalizza output del layer precedente
    layers.Activation('relu'),
    layers.Dropout(0.3),

    layers.Dense(64),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),

    layers.Dense(num_classes, activation='softmax')
])
```

#### Learning Rate Scheduling

```python
# Riduce learning rate quando val_loss si stabilizza
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,        # Riduce LR della metà
    patience=5,
    min_lr=1e-7,
    verbose=1
)

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=100,
    callbacks=[early_stopping, reduce_lr]
)
```

### 9. Salvare e caricare il modello

```python
# Salva modello completo
model.save('my_model.h5')

# Salva solo i pesi
model.save_weights('model_weights.h5')

# Carica modello
loaded_model = keras.models.load_model('my_model.h5')

# Carica solo i pesi
model.load_weights('model_weights.h5')
```

### 10. Hyperparameter Tuning

#### Grid Search con Keras

```python
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

def create_model(neurons=64, dropout_rate=0.3, learning_rate=0.001):
    model = models.Sequential([
        layers.Input(shape=(X_train_scaled.shape[1],)),
        layers.Dense(neurons, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Wrap model
model = KerasClassifier(build_fn=create_model, verbose=0)

# Define hyperparameters
param_grid = {
    'neurons': [32, 64, 128],
    'dropout_rate': [0.2, 0.3, 0.5],
    'learning_rate': [0.001, 0.0001],
    'batch_size': [16, 32],
    'epochs': [50]
}

# Grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=2)
grid_result = grid.fit(X_train_scaled, y_train)

print(f"Best: {grid_result.best_score_:.4f} using {grid_result.best_params_}")
```

### 11. Checklist finale

Prima di considerare il modello completo, verifica:

- [ ] **Preprocessing**: dati scalati correttamente?
- [ ] **Split**: train/val/test separati correttamente?
- [ ] **Architettura**: numero di layer e neuroni appropriato?
- [ ] **Overfitting**: val_loss stabile o in crescita?
- [ ] **Underfitting**: accuracy troppo bassa su entrambi i set?
- [ ] **Metriche**: accuracy, precision, recall, F1-score valutati?
- [ ] **Confusion Matrix**: errori sistematici su alcune classi?
- [ ] **Callbacks**: early stopping e checkpoint configurati?
- [ ] **Test finale**: modello valutato su test set mai visto?

### 12. Troubleshooting comuni

| Problema              | Causa probabile           | Soluzione                        |
| --------------------- | ------------------------- | -------------------------------- |
| Loss = NaN            | Learning rate troppo alta | Riduci LR a 0.0001               |
| Accuracy = 0 o random | Dati non normalizzati     | Applica StandardScaler           |
| Val_accuracy oscilla  | Batch size troppo piccolo | Aumenta a 64-128                 |
| Training molto lento  | Batch size troppo piccolo | Aumenta batch_size               |
| Memory error          | Batch size troppo grande  | Riduci batch_size                |
| Overfitting immediato | Troppi parametri          | Aggiungi Dropout, riduci neuroni |

---

## Risorse utili

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Guide](https://keras.io/guides/)
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [Papers With Code](https://paperswithcode.com/)
