import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

df = pd.read_csv(url, names=column_names)

# Paso 1: Análisis exploratorio de datos (EDA)
def plot_correlations(df):
    corr_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.show()

plot_correlations(df)

# Paso 2: Limpieza de datos
# Reemplazar valores nulos por la media o eliminar filas con valores nulos (si existieran)
imputer = SimpleImputer(strategy="mean")
df.iloc[:, :-1] = imputer.fit_transform(df.iloc[:, :-1])

# Paso 3: Codificación de variables categóricas (en este caso, la clase)
label_encoder = LabelEncoder()
df['class'] = label_encoder.fit_transform(df['class'])

# Paso 4: Normalización de las características numéricas (sepal_length, sepal_width, etc.)
scaler = StandardScaler()
df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']] = scaler.fit_transform(
    df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
)

# Paso 5: Dividir el dataset en entrenamiento y prueba (si es necesario para otros modelos)
X = df.drop(columns='class')
y = df['class']

# Dividir en conjunto de entrenamiento y prueba (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 6: Guardar el dataset procesado
df.to_csv('data/processed/iris_processed.csv', index=False)

# Guardar los datos de entrenamiento y prueba
X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)

print("Dataset procesado y guardado en 'data/processed/iris_processed.csv'")
