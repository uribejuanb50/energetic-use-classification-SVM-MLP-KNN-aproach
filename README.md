# energetic-use-classification-SVM-MLP-KNN-aproach

steel_load_classification/
├── data/
│   ├── raw/                      # CSV original (no se toca)
│   └── processed/                # Splits y datasets procesados (versionados o en .gitignore)
├── notebooks/
│   ├── 01_eda.ipynb              # Exploración inicial, distribuciones, leakage check
│   └── 02_results_analysis.ipynb # Análisis post-experimento
├── src/
│   ├── config.py                 # Rutas, semillas, hiperparámetros, paths
│   ├── data/
│   │   ├── loader.py             # Lee CSV crudo, valida esquema
│   │   ├── preprocessor.py       # Encoding, escalado, manejo de cíclicas
│   │   └── splitter.py           # Split TEMPORAL train/val/test
│   ├── features/
│   │   └── engineering.py        # Cíclicas (sin/cos), interacciones si aplica
│   ├── models/
│   │   ├── base.py               # Interfaz abstracta: fit, predict, predict_proba, save, load
│   │   ├── svm_model.py          # Wrapper de sklearn con grid search
│   │   └── mlp_model.py          # Wrapper de TF/Keras con callbacks (EarlyStopping, ReduceLROnPlateau)
│   ├── evaluation/
│   │   ├── metrics.py            # Accuracy, F1 macro, recall por clase, ROC-AUC OvR
│   │   ├── visualization.py      # Matriz de confusión, curvas ROC, curvas de aprendizaje
│   │   └── statistical.py        # Test de McNemar y/o t-test pareado sobre folds
│   └── experiments/
│       ├── run_svm.py            # Entrena SVM con su grid search
│       ├── run_mlp.py            # Entrena MLP con su búsqueda de arquitectura
│       └── compare.py            # Compara los dos mejores modelos estadísticamente
├── reports/
│   ├── figures/                  # Gráficas para el PPT
│   └── metrics/                  # JSON/CSV con resultados (para reproducibilidad)
├── tests/
│   └── test_preprocessing.py     # Mínimo: que el preprocesamiento no rompa con datos vacíos/atípicos
├── models/
├── requirements.txt
├── README.md
└── main.py

### src/data

## src/data/loader.py
Propósito. Este módulo es el portero del proyecto. Recibe el CSV crudo del disco y devuelve un DataFrame que el resto del sistema puede asumir válido. Si algo está mal con el archivo (columnas faltantes, fechas que no parsean, nulls inesperados), falla aquí, no a las 3 horas de entrenamiento. La filosofía: fail fast, fail loud.

## src/data/preprocessor.py
Propósito. Limpiar y codificar columnas para que los modelos las puedan consumir. Es importante lo que este módulo no hace:
No filtra filas (no es el splitter).
No crea features nuevas (eso es engineering.py).
No escala valores numéricos (eso es después del split, fit solo en train).
No toca la columna date (se necesita para el split, se descarta más adelante en main.py).
Es decir, preprocessor transforma columnas existentes una a una. Limpia, no inventa.

## src/data/splitter.py
Especificación...
Propósito del módulo. Particionar el dataset respetando el orden temporal, ofrecer una utilidad para submuestreo estratificado (que solo se usará en el grid search de SVM), y validar que los splits cumplan los invariantes que necesitamos. Si este módulo está mal, todo el experimento queda inválido aunque los modelos entrenen bien.

## src/features/engineering.py
Propósito. Crear features derivadas a partir de las columnas existentes. Específicamente: codificar las variables cíclicas (NSM y Day_of_week) como pares seno/coseno para que los modelos puedan capturar la cercanía temporal real (23:59 <-> 00:01, domingo <-> sábado).
Este módulo no limpia (eso es preprocessor), no escala (eso es después del split), y no toca el target. Solo transforma features cíclicas.