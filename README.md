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
├── requirements.txt
├── README.md
└── main.py