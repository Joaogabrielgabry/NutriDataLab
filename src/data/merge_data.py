"""
merge_data.py
Integra os 4 DataFrames em um único dataset consolidado.
"""

import pandas as pd


def merge_all(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Faz o merge dos 4 datasets pelo patient_id, diet_id e nutritionist_id.
    Retorna um DataFrame unificado com todas as features.
    """
    patients      = data["patients"]
    diets         = data["diets"]
    nutritionists = data["nutritionists"]
    results       = data["results"]

    # patients + results
    df = patients.merge(results, on="patient_id", how="inner")

    # + diets
    df = df.merge(diets, on="diet_id", how="left", suffixes=("", "_diet"))

    # + nutritionists
    df = df.merge(nutritionists, on="nutritionist_id", how="left", suffixes=("", "_nutr"))

    print(f"✅ Dataset consolidado: {df.shape[0]} linhas × {df.shape[1]} colunas")
    return df