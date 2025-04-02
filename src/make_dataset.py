# src/make_dataset.py

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import os

# Función para cargar y preprocesar el dataset Iris
def load_data():
    # Cargar el dataset Iris desde sklearn
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['species'] = iris.target_names[iris.target]
    
    return data

# Preprocesamiento de datos: Escalado de características
def preprocess_data(data):
    # Separar características (X) y etiquetas (y)
    X = data.drop('species', axis=1)
    y = data['species']
    
    # Escalar las características (aunque los árboles de decisión no se ven muy afectados por esto)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

# Guardar los datos preprocesados
def save_processed_data(X, y):
    if not os.path.exists('data/processed'):
        os.makedirs('data/processed')
    
    # Guardar X y y como archivos CSV
    pd.DataFrame(X, columns=['sepal length', 'sepal width', 'petal length', 'petal width']).to_csv('data/processed/X.csv', index=False)
    y.to_csv('data/processed/y.csv', index=False)

if __name__ == '__main__':
    data = load_data()
    X, y = preprocess_data(data)
    save_processed_data(X, y)
