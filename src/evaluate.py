# src/evaluate.py

import joblib
import pandas as pd
from sklearn.metrics import classification_report

# Cargar el modelo entrenado
model = joblib.load('models/decision_tree_model.pkl')

# Cargar los datos de prueba
X_test = pd.read_csv('data/processed/X.csv')
y_test = pd.read_csv('data/processed/y.csv')

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_test, y_test, test_size=0.2, random_state=42)

# Realizar las predicciones
y_pred = model.predict(X_test)

# Generar un reporte de clasificación
report = classification_report(y_test, y_pred)

# Guardar el reporte de clasificación
with open('data/scores/classification_report.txt', 'w') as f:
    f.write(report)

print("Reporte de clasificación guardado en 'data/scores/classification_report.txt'")
