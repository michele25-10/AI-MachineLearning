# Le malattie cardiovascolari (CVD) sono la principale causa di morte
# a livello globale, con una stima di 17,9 milioni di decessi ogni anno,
# pari al 31% di tutte le morti nel mondo.
#
# L’insufficienza cardiaca è una conseguenza comune delle CVD e questo
# dataset contiene 12 caratteristiche che possono essere utilizzate per
# prevedere la mortalità dovuta a insufficienza cardiaca.
#
# La maggior parte delle malattie cardiovascolari può essere prevenuta 
# affrontando i fattori di rischio comportamentali, come:
# - uso di tabacco
# - dieta non sana e obesità
# - inattività fisica
# - uso dannoso di alcol
# attraverso strategie di prevenzione su larga scala.
# 
# Le persone con malattie cardiovascolari o ad alto rischio cardiovascolare 
# (a causa della presenza di uno o più fattori di rischio come ipertensione,
# diabete, iperlipidemia o malattie già diagnosticate) necessitano di una
# diagnosi precoce e di una gestione adeguata, ambiti in cui un modello di 
# machine learning può offrire un valido supporto.
# =========================================================================

import os
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


file_path = "../data/heart_failure_clinical_records_dataset.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError("CSV inesistente")

df = pd.read_csv(file_path, na_values=["NA", "-", "N/A", "NaN", ""], nrows=1000000)

print("\nInformazioni sul dataframe:")
print(df.info())
print(df.head())

        
# droppo le colonne delle categorie true e false che non mi interessano
df = df.dropna(axis=1, how="all")
df = df.dropna(axis=1, how="any")

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
target = "DEATH_EVENT"

correlation_matrix = df[num_cols].corr()
plt.figure(figsize=(12, 12))
sns.heatmap(correlation_matrix, fmt=".2f", linewidths=1.5, linecolor="white", annot=True, cmap="coolwarm")
plt.show()

if target in num_cols:
    num_cols.remove(target) 

num_cols.remove("sex")
num_cols.remove("smoking")
num_cols.remove("diabetes")


X = df[num_cols]
y = df[target]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Score accuracy: {accuracy:.2f}")

plt.figure(figsize=(12, 12))
plot_tree(model, max_depth=2)
plt.show()

conf_matrix = confusion_matrix(y_test ,y_pred)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=sorted(y.unique()))
disp.plot(cmap="Blues")
plt.show()