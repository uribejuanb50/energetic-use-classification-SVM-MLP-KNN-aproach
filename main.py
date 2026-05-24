import src.data.loader as loader
import src.data.preprocessor as preprocessor
import src.data.splitter as splitter
import src.features.engineering as engineering

from src.models.base import BaseModel
from src.models.svm_model import SVMmodel
from src.config import (VALORES_FUGA, 
                        MESES_ENTRENAMIENTO, 
                        MESES_EVALUACION,
                        SEMILLA,
                        SVM_GRID,
                        SVM_GRIDSEARCH_SUBSAMPLE
                        )

import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report


def main():
    

    df = loader.leer_csv(Path("data/raw/Steel_industry_data.csv"))

    df, mapeo_target = preprocessor.preprocesar(df, VALORES_FUGA)

    df = engineering.aplicar_caracteristicas_ciclicas(df)

    train, evaluar, test = splitter.dividir_temporal(df, "date", MESES_ENTRENAMIENTO, MESES_EVALUACION)
    reporte = splitter.verificar_split(train, evaluar, test, "Load_Type", "date")

    print(f"Reporte de los splits:\n{reporte}")

    #Separar los sets em caracteristicas y objetivos
    objetivo_train = train["Load_Type"]
    caractaristicas_train = train.drop(columns = ["date", "Load_Type"])

    objetivo_evaluar = evaluar["Load_Type"]
    caracteristicas_evaluar = evaluar.drop(columns = ["date", "Load_Type"])

    objetivo_test = test["Load_Type"]
    caracteristicas_test = test.drop(columns = ["date", "Load_Type"])

    scaler = StandardScaler()
    caracteristicas_train_escalado = scaler.fit_transform(caractaristicas_train)
    caracteristicas_evaluar_escalado = scaler.transform(caracteristicas_evaluar)
    caracteristicas_test_escalado = scaler.transform(caracteristicas_test)

    svm_model : BaseModel = SVMmodel({"SVM_GRID" : SVM_GRID, 
                                       "tam_subsampleo" : SVM_GRIDSEARCH_SUBSAMPLE, 
                                       "semilla" : SEMILLA})
    
    svm_model.fit(caracteristicas_train_escalado, objetivo_train, "Load_Type")

    objetivo_predict = svm_model.predict(caracteristicas_test_escalado)
    print(f"[Main] f1 report: {f1_score(objetivo_test, objetivo_predict, average = 'macro'):.4f} ")


if __name__ == "__main__":
    main()