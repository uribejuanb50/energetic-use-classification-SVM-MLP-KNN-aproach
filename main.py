import src.data.loader as loader
import src.data.preprocessor as preprocessor
import src.data.splitter as splitter
import src.features.engineering as engineering
from src.config import VALORES_FUGA, MESES_ENTRENAMIENTO, MESES_EVALUACION

import json
from pathlib import Path


def main():
    

    df = loader.leer_csv(Path("data/raw/Steel_industry_data.csv"))

    df, mapeo_target = preprocessor.preprocesar(df, VALORES_FUGA)

    df = engineering.aplicar_caracteristicas_ciclicas(df)

    train, evaluar, test = splitter.dividir_temporal(df, "date", MESES_ENTRENAMIENTO, MESES_EVALUACION)
    reporte = splitter.verificar_split(train, evaluar, test, "Load_Type", "date")

    objetivo_train = train["Load_Type"]
    caractaristicas_train = train.drop(columns = ["date", "Load_Type"])

if __name__ == "__main__":
    main()