# ML-Engineering-Project
ML Engineering Project

# Proyecto de Machine Learning - Árboles de Decisión

## 📌 Descripción
Este proyecto implementa un modelo de **Árbol de Decisión** para clasificación, basado en el dataset **Iris**. El código sigue una estructura modular para facilitar su uso y mantenimiento.

## 📂 Estructura del Proyecto
```
├── data
│   ├── raw            # Dataset original sin procesar
│   ├── processed      # Dataset preprocesado listo para entrenamiento
│   ├── scores         # Resultados de métricas de evaluación
│
├── models             # Modelos entrenados y serializados
│
├── notebooks          # Análisis exploratorio y experimentos
│
├── src                # Código fuente
│   ├── make_dataset.py # Preprocesamiento de datos
│   ├── train.py        # Entrenamiento del modelo
│   ├── evaluate.py     # Evaluación del modelo
│   ├── predict.py      # Predicciones con nuevos datos
│
├── requirements.txt    # Dependencias del proyecto
├── setup.py            # Instalación del proyecto
├── LICENSE             # Licencia
├── README.md           # Documentación del proyecto
```

## ⚙️ Instalación
1. Clonar el repositorio:
   ```bash
   git clone https://github.com/usuario/proyecto-ml-decision-tree.git
   cd proyecto-ml-decision-tree
   ```
2. Crear un entorno virtual e instalar dependencias:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows usar: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## 🚀 Uso del Proyecto
### 1️⃣ Preprocesar los datos
```bash
python src/make_dataset.py
```
### 2️⃣ Entrenar el modelo
```bash
python src/train.py
```
### 3️⃣ Evaluar el modelo
```bash
python src/evaluate.py
```
### 4️⃣ Realizar predicciones
```bash
python src/predict.py --input "data/input.csv"
```

## 🛠️ Requisitos
- Python 3.8+
- Scikit-learn
- Pandas
- NumPy

## ✨ Contribuciones
Si deseas contribuir, por favor crea un **fork** del repositorio, realiza tus cambios y envía un **pull request**.

## 📜 Licencia
Este proyecto está bajo la licencia MIT. Para más detalles, revisa el archivo LICENSE.
