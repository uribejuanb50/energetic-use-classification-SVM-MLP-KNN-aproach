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