import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# ID3
from sklearn.tree import DecisionTreeClassifier

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# CARGA DEL DATASET
# ============================================

df = pd.read_csv("heart_disease.csv")  

print("Primeras filas del dataset:")
print(df.head())        

print("\nInformación del dataset:")
print(df.info())

print("\nDescripción estadística:")
print(df.describe())   

print(f"\nDataset cargado correctamente con {df.shape[0]} filas y {df.shape[1]} columnas.")

# ============================================
# PREPROCESAMIENTO GENERAL
# ============================================

# Variable objetivo
y = df["target_binary"]

# Variables predictoras 
X = df.drop(columns=["target_binary", "num"], errors="ignore")

# División entrenamiento/prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Escalado de características
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nPreprocesamiento completado correctamente.")

# ============================================
# ID3 — ENTRENAMIENTO DEL MODELO
# ============================================

id3 = DecisionTreeClassifier(
    criterion="entropy",   # ID3 usa entropía
    max_depth=5,
    random_state=42
)

id3.fit(X_train_scaled, y_train)

# ============================================
# PREDICCIONES
# ============================================

y_pred = id3.predict(X_test_scaled)
y_prob = id3.predict_proba(X_test_scaled)[:, 1]

# ============================================
# MÉTRICAS
# ============================================

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

# ============================================
# MATRIZ DE CONFUSIÓN
# ============================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusión — ID3 (Entropy)")
plt.xlabel("Predicción")
plt.ylabel("Verdadero")
plt.show()

# ============================================
# CURVA ROC
# ============================================

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("Curva ROC — ID3")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
