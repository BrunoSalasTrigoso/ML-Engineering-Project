from setuptools import setup, find_packages

setup(
    name='ml-decision-tree',  # Nombre del proyecto
    version='0.1',  # Versión del proyecto
    packages=find_packages(),  # Esto buscará todos los paquetes en el proyecto
    install_requires=[  # Aquí se listan las dependencias del proyecto
        'numpy>=1.21.0',  # Para la manipulación numérica
        'pandas>=1.3.0',  # Para la manipulación de datos
        'scikit-learn>=0.24.0',  # Para entrenar y usar modelos de Machine Learning (como Árboles de Decisión)
        'joblib>=1.0.0',  # Para la serialización y deserialización de modelos (guardar y cargar)
        'matplotlib>=3.4.0',  # Para visualización de datos (si decides agregar gráficos en los notebooks)
        'seaborn>=0.11.0'  # Para una visualización estadística más avanzada (opcional, pero útil en EDA)
    ],
    classifiers=[  # Información adicional sobre el paquete
        'Programming Language :: Python :: 3',  # Python 3
        'License :: OSI Approved :: MIT License',  # Licencia MIT
        'Operating System :: OS Independent',  # Compatible con sistemas operativos independientes
    ],
    python_requires='>=3.8',  # Aseguramos que se usa Python 3.8 o superior
)
