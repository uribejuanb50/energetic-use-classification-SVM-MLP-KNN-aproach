import src.data.loader as loader
import src.data.preprocessor as preprocessor
import src.data.splitter as splitter
import src.features.engineering as engineering

from src.models.base import BaseModel
from src.models.svm_model import SVMmodel
from src.models.mlp_model import MLPmodel
from src.config import (
                        VALORES_FUGA, 
                        MESES_ENTRENAMIENTO, 
                        MESES_EVALUACION,
                        SEMILLA,
                        SVM_GRID,
                        SVM_GRIDSEARCH_SUBSAMPLE,
                        MODELS,
                        FIGURES
                        )

import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression


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
    print(f"[Main] f1 report: {f1_score(objetivo_test, objetivo_predict, average = 'macro'):.4f}\n\n\n ")

    mlp_model : BaseModel = MLPmodel(n_features = caracteristicas_train_escalado.shape[1])
    mlp_model.fit(caracteristicas_train_escalado, objetivo_train,
                  caracteristicas_evaluar_escalado, objetivo_evaluar)
    mlp_model.save( MODELS / "mlp_model.keras")

    for nombre, modelo in [("SVM", svm_model), ("MLP", mlp_model)]:
        print(f"{nombre}========================================")
        objetivo_predict = modelo.predict(caracteristicas_test_escalado)
        print(f"[Main] f1 report: {f1_score(objetivo_test, objetivo_predict, average = 'macro'):.4f}\n\n\n ")
        print(f"[Main] reporte:\n{classification_report(objetivo_test, objetivo_predict, target_names = mapeo_target.keys())} ")


    # Concatene los tres splits con una columna que diga cuál
    train["split"] = "train"
    evaluar["split"] = "val"
    test["split"] = "test"
    todo = pd.concat([train, evaluar, test])
    todo["mes"] = todo["date"].dt.month

    # Distribución del target por mes
    print(todo.groupby("mes")["Load_Type"].value_counts(normalize=True).unstack())

    # Distribución de features clave por split
    features_clave = ["Lagging_Current_Reactive.Power_kVarh", "Lagging_Current_Power_Factor"]
    for f in features_clave:
        todo.boxplot(column=f, by="split")
        plt.title(f)
        plt.show()
        plt.savefig(FIGURES / f"{f}_image.png")

   

    dummy = DummyClassifier(strategy='stratified', random_state=SEMILLA)
    dummy.fit(caracteristicas_train_escalado, objetivo_train)
    y_pred_dummy = dummy.predict(caracteristicas_test_escalado)
    print(f"Dummy F1 macro: {f1_score(objetivo_test, y_pred_dummy, average='macro'):.4f}")

    logreg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=SEMILLA)
    logreg.fit(caracteristicas_train_escalado, objetivo_train)
    y_pred_logreg = logreg.predict(caracteristicas_test_escalado)
    print(f"LogReg F1 macro: {f1_score(objetivo_test, y_pred_logreg, average='macro'):.4f}")


if __name__ == "__main__":
    main()