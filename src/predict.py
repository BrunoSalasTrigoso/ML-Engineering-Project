# src/predict.py

import pandas as pd
import joblib
import argparse

# Función para predecir con el modelo entrenado
def predict(input_file):
    # Cargar el modelo entrenado
    model = joblib.load('models/decision_tree_model.pkl')
    
    # Cargar los nuevos datos
    input_data = pd.read_csv(input_file)
    X_input = input_data.drop('species', axis=1)  # Asegurarse de que la columna 'species' no esté incluida
    
    # Realizar la predicción
    predictions = model.predict(X_input)
    
    # Guardar las predicciones
    input_data['predictions'] = predictions
    input_data.to_csv('data/scores/predictions.csv', index=False)
    print(f"Predicciones guardadas en 'data/scores/predictions.csv'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Realizar predicciones con el modelo de árbol de decisión")
    parser.add_argument('--input', type=str, help="Archivo de datos para hacer predicciones", required=True)
    args = parser.parse_args()

    predict(args.input)
