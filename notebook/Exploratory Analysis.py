# Importar las librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Cargar el dataset Iris de sklearn
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Vista general de los primeros registros
print("Primeros registros del dataset:")
display(data.head())

# Información general sobre los datos
print("Información general del dataset:")
print(data.info())

# Descripción estadística de las variables numéricas
print("Descripción estadística de las variables numéricas:")
print(data.describe())

# Verificar la cantidad de registros por cada clase (species)
print("Distribución de las clases:")
print(data['species'].value_counts())

# Visualización de la distribución de las características numéricas por clase
plt.figure(figsize=(12, 8))

# Histograma para cada variable
for i, col in enumerate(data.columns[:-1]):
    plt.subplot(2, 2, i+1)
    sns.histplot(data=data, x=col, hue='species', kde=True, multiple='stack')
    plt.title(f'Distribución de {col}')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')

plt.tight_layout()
plt.show()

# Matriz de correlación entre las características numéricas
correlation_matrix = data.drop(columns='species').corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matriz de correlación entre las características')
plt.show()

# Boxplots para visualizar la distribución de cada característica por especie
plt.figure(figsize=(12, 8))

for i, col in enumerate(data.columns[:-1]):
    plt.subplot(2, 2, i+1)
    sns.boxplot(data=data, x='species', y=col)
    plt.title(f'Boxplot de {col} por especie')

plt.tight_layout()
plt.show()

# Pairplot de las variables para ver la relación entre ellas
sns.pairplot(data, hue='species')
plt.suptitle("Pairplot de las características", y=1.02)
plt.show()

# Análisis de valores faltantes (aunque Iris no tiene)
missing_data = data.isnull().sum()
print("Valores faltantes por columna:")
print(missing_data)

# Normalización de las características numéricas
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data.drop(columns='species')), columns=data.columns[:-1])
data_scaled['species'] = data['species']

# Mostrar las primeras filas después de la normalización
print("Primeras filas después de la normalización:")
display(data_scaled.head())
