import joblib
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from src.data.splitter import submuestreo_estratificado
from src.models.base import BaseModel
from src.config import SVM_GRIDSEARCH_SUBSAMPLE

class SVMmodel(BaseModel) :

    def __init__ (self, config : dict, col_objetivo = "Load_Type") :
        self.svm_grid = config["SVM_GRID"]
        self.tam_subsampleo = config["tam_subsampleo"]
        self.semilla = config["semilla"]
        self.col_objetivo = col_objetivo

        return 
    
    def fit(self, x_train, y_train, x_evaluar = None, y_evaluar = None) :

        if not isinstance(y_train, pd.Series):

            y_train_copia = pd.Series(y_train, name=self.col_objetivo)

        else:
            y_train_copia = y_train.copy()
            y_train_copia.name = self.col_objetivo

        if not isinstance(x_train, pd.DataFrame):
            columnas_originales = getattr(x_train, "columns", None)
            x_train_copia = pd.DataFrame(x_train, index=y_train_copia.index, columns=columnas_originales)

        else:
            x_train_copia = x_train.copy()

        df_train_temp = pd.concat([x_train_copia, y_train_copia], axis = 1)

        train_submuestrado = submuestreo_estratificado(df_train_temp, self.col_objetivo, self.tam_subsampleo, self.semilla)

        objetivos_train_submuestreado = train_submuestrado[self.col_objetivo]
        caracteristicas_train_submuestreado = train_submuestrado.drop(columns = [self.col_objetivo])

        svm_base = SVC(class_weight = "balanced", random_state = self.semilla)

        grid = GridSearchCV(
                            estimator = svm_base,
                            param_grid = self.svm_grid,
                            scoring = "f1_macro",
                            cv = 3,
                            verbose = 3
                            )
        
        grid.fit(caracteristicas_train_submuestreado, objetivos_train_submuestreado)
        
        mejor_parametro = grid.best_params_

        self.model = SVC(**mejor_parametro, probability = True, class_weight = "balanced", )

        self.model.fit(x_train, y_train)

        print("[Svm_model] Entrenamiento completado con éxito.")
        print(f"[Svm_model] El mejor parámetro fue: {mejor_parametro}")
        print(f"[Svm_model] Mejor CV: {grid.best_score_:.4f}")

        return 
    
    def predict(self, X) :
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def save(self, path) :
        joblib.dump(self.model, path)

    def load(self, path) :
        self.model = joblib.load(path)






