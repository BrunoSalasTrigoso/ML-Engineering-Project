# Importar las librerías necesarias
import joblib
import pandas as pd
from sklearn.datasets import load_iris

# Cargar el modelo entrenado
model = joblib.load('models/decision_tree_model.pkl')

# Cargar un nuevo conjunto de datos para predecir (usaremos el conjunto completo de Iris)
iris = load_iris()
X_new = iris.data  # Puedes cambiar esto por nuevos datos si lo deseas

# Hacer predicciones con el modelo cargado
predictions = model.predict(X_new)

# Mostrar las predicciones
predictions_df = pd.DataFrame(predictions, columns=['Predicted Species'])
predictions_df['Predicted Species'] = predictions_df['Predicted Species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print(predictions_df.head())
