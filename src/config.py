from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
# Si tu archivo está en src/config.py, parents[1] sube a la raíz del proyecto.
# Si la estructura cambia, ajusta este número.
DATA_RAW = RAIZ / "data" / "raw" / "Steel_industry_data.csv"
# Ajuste el nombre exacto del CSV al que descargó de UCI.
DATA_PROCESSED = RAIZ / "data" / "processed"
REPORTS = RAIZ / "reports"
FIGURES = REPORTS / "figures"
METRICS = REPORTS / "metrics"
MODELS = RAIZ / "models"


VALORES_FUGA = ["Usage_kWh", "CO2(tCO2)"]
SEMILLA = 42


MESES_ENTRENAMIENTO = 9
MESES_EVALUACION = 1


SVM_GRIDSEARCH_SUBSAMPLE = 8000
SVM_GRID = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", 0.01, 0.1, 1],
    "kernel": ["rbf"],
}


MLP_CONFIG = {
    "hidden_layers": [64, 32],     # Neuronas por capa oculta
    "dropout": [0.3, 0.2],         # Dropout después de cada capa
    "lr": 1e-3,                    # Learning rate inicial para Adam
    "batch_size": 128,
    "epochs": 200,                 # EarlyStopping va a cortar antes
    "patience": 10,                # Epochs sin mejorar antes de parar
}
N_CLASES = 3