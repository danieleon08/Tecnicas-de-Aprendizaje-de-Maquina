# =========================================================
# IMPORTS Y LIBRERÍAS NECESARIAS
# =========================================================

# Manejo de datos
import numpy as np
import pandas as pd

# Preprocesamiento y ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# SCRIPT 2 — CARGA DEL CSV Y VISUALIZACIÓN
# =========================================================

# Cargar dataset (asegúrate que el archivo esté en la misma carpeta)
df = pd.read_csv("heart_disease.csv")

# Mostrar las primeras filas
print("Primeras filas del dataset:")
print(df.head().to_string())

# Mostrar información general
print("\nInformación del dataset:")
print(df.info())

# Mostrar estadísticas
print("\nDescripción estadística:")
print(df.describe().to_string())

# Confirmar número de filas y columnas
print(f"\nDataset cargado correctamente con {df.shape[0]} filas y {df.shape[1]} columnas.")


# =========================================================
# SCRIPT 3 — ENTRENAMIENTO SVM Y GRÁFICAS COMPLETAS
# =========================================================

# -------- Selección de variables --------
y = df["target_binary"]
X = df.drop(columns=["target_binary", "num"], errors="ignore")

# -------- División train/test --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------- Escalado --------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------- Entrenamiento SVM --------
svm = SVC(kernel="rbf", probability=True)
svm.fit(X_train_scaled, y_train)

# -------- Predicciones --------
y_pred = svm.predict(X_test_scaled)
y_prob = svm.predict_proba(X_test_scaled)[:, 1]

# -------- Reporte --------
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))


# =========================================================
# GRÁFICAS
# =========================================================

# -------- Matriz de confusión --------
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusión — SVM")
plt.xlabel("Predicción")
plt.ylabel("Verdadero")
plt.tight_layout()
plt.show()

# -------- Curva ROC --------
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC — SVM")
plt.legend()
plt.tight_layout()
plt.show()
