import joblib
import numpy as np

from pathlib import Path

#keras es de tensorflow.keras, pero solo así no tira la fastidiosa advertencia JAJAJ
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

from src.models.base import BaseModel
from src.config import MLP_CONFIG, N_CLASES

class MLPmodel(BaseModel):

    def __init__(self, n_features, n_clases = N_CLASES, config = MLP_CONFIG) :
        self.n_features = n_features
        self.n_clases = n_clases
        self.config = config
        self.history = None
        self.model = self._construir_modelo()

        return 
    
    def _construir_modelo(self) -> Sequential:
        modelo = Sequential()

        #modelo.add(Input(shape = self.n_features,))

        modelo.add(Input(shape=(self.n_features,)))

        for neuronas, dropout in zip(self.config["hidden_layers"], self.config["dropout"]) :
            modelo.add(Dense(neuronas, activation = "relu"))
            modelo.add(Dropout(dropout))

        modelo.add(Dense(self.n_clases, activation = "softmax"))

        modelo.compile(optimizer = Adam(learning_rate = self.config["lr"]),
                       loss = "sparse_categorical_crossentropy",
                       metrics=["accuracy"],
                       )
        return modelo

    def fit(self, x_train, y_train, x_evaluar = None, y_evaluar = None) :

        clases = np.unique(y_train)
        pesos = compute_class_weight(class_weight = "balanced",
                                     classes = clases,
                                     y = y_train)
        
        clases_con_pesos = dict(zip(clases, pesos))

        print(f"[Mlp_model] aquí claramente está cada clase con su peso jiji\n{clases_con_pesos}")

        parada_temprana = EarlyStopping(monitor = "val_loss",
                                   patience = self.config["patience"],
                                   restore_best_weights = True,
                                   verbose = 1)
        reducir_ritmo_aprendizaje = ReduceLROnPlateau(monitor = "val_loss",
                                                      patience = self.config["patience"] // 2,
                                                      factor = 0.5,
                                                      min_lr = 1e-6,
                                                      verbose = 1)
        
        data_evaluar = (x_evaluar, y_evaluar) if x_evaluar is not None else None 

        history = self.model.fit(x_train,
                                 y_train,
                                 validation_data = data_evaluar,
                                 epochs = self.config["epochs"],
                                 batch_size = self.config["batch_size"],
                                 class_weight = clases_con_pesos,
                                 callbacks = [parada_temprana, reducir_ritmo_aprendizaje],
                                 verbose = 2)

        self.history = history.history

        print(f"[Mlp_model] Entrenamiento finalizado... Épocas efectivas: {len(self.history['loss'])} ")
        return  
    
    def predict(self, X) -> np.ndarray :
        probabilidades = self.model.predict(X, verbose = 0)
        predicciones = np.argmax(probabilidades, axis = 1)
        
        return predicciones

    def predict_proba(self, X) -> np.ndarray :
        return self.model.predict(X, verbose = 0)

    def save(self, path : Path) :
        self.model.save(path)
        print(f"[Mlp_model] salvado en {path}")

        return

    def load(self, path) :
        self.model = load_model(path) 
        print(f"[Mlp] modelo cargado desde {path}")

        return
             
