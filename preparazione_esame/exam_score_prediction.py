# Informazioni sul dataset

# Questo dataset fornisce una rappresentazione ampia e realistica dei vari fattori che contribuiscono 
# alle prestazioni degli studenti agli esami. Contiene 20.000 record, ognuno dei quali descrive il 
# comportamento accademico di uno studente, le sue abitudini di studio, le routine di vita e le 
# condizioni d’esame. Queste variabili aiutano collettivamente a comprendere come diversi aspetti della
# vita quotidiana e dell’ambiente di apprendimento influenzino i risultati agli esami.
# 
# Il dataset include dettagli come: ore di studio giornaliere, percentuale di frequenza alle lezioni,
# durata e qualità del sonno, disponibilità di internet, metodo di studio, valutazione delle strutture
# dell’istituto e livello di difficoltà dell’esame. Questi fattori riflettono un ampio spettro di 
# influenze comunemente osservate nei contesti educativi. Il punteggio d’esame (0–100) è ottenuto tramite
# una formula ponderata che riproduce modelli realistici di rendimento scolastico.

# Trova un modello per predire i risultati di un test

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

file_path = "../data/Exam_Score_Prediction.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError("CSV non trovato")

df = pd.read_csv(file_path, na_values=['-', 'NA', 'N/A', 'NaN'], nrows=10000000)


print("\nDataset information: ")
print(df.info())

df.drop(columns=["student_id"])

df["internet_access"] = df["internet_access"].replace({
    'yes': 1, 
    'no': 0
})

print(f"Tipologia di dati internet_accecss: {df["internet_access"].unique()}")