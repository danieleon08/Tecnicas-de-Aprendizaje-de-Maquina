# Manejo de datos
import numpy as np
import pandas as pd

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc
)

# Red Neuronal
from sklearn.neural_network import MLPClassifier

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# CARGA DEL DATASET
# =========================================================

# Cambia esta ruta por la ruta en tu PC
df = pd.read_csv("heart_disease.csv")

# Mostrar las primeras filas
print("Primeras filas del dataset:")
print(df.head())

# Mostrar información general
print("\nInformación del dataset:")
print(df.info())

# Mostrar estadísticas
print("\nDescripción estadística:")
print(df.describe())

# Confirmar estructura del dataset
print(f"\nDataset cargado correctamente con {df.shape[0]} filas y {df.shape[1]} columnas.")


# =========================================================
# PREPROCESAMIENTO GENERAL
# =========================================================

# Variable objetivo
y = df["target_binary"]

# Variables predictoras
X = df.drop(columns=["target_binary", "num"], errors="ignore")

# División del dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nPreprocesamiento completado correctamente.")


# =========================================================
# ENTRENAR MLP Y GRAFICAR RESULTADOS
# =========================================================

mlp = MLPClassifier(
    hidden_layer_sizes=(50,30),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)

# Entrenar modelo
mlp.fit(X_train_scaled, y_train)

# Predicciones
y_pred = mlp.predict(X_test_scaled)
y_prob = mlp.predict_proba(X_test_scaled)[:, 1]

# Métricas
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\n=== Reporte de Clasificación ===")
print(classification_report(y_test, y_pred))


# =========================================================
# MATRIZ DE CONFUSIÓN
# =========================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusión — MLP")
plt.xlabel("Predicción")
plt.ylabel("Verdadero")
plt.tight_layout()
plt.show()


# =========================================================
# CURVA ROC
# =========================================================

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC — MLP")
plt.legend()
plt.tight_layout()
plt.show()

