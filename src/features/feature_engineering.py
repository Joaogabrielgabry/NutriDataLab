"""
feature_engineering.py
Criação de features derivadas para enriquecer o dataset.
"""

import pandas as pd
import numpy as np


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona features engenheiradas ao DataFrame consolidado (pré-encode).
    Deve ser chamado ANTES do preprocess().
    """
    df = df.copy()

    # IMC categórico
    df["bmi_category"] = pd.cut(
        df["bmi"],
        bins=[0, 18.5, 24.9, 29.9, 40, 999],
        labels=["Abaixo do Peso", "Normal", "Sobrepeso", "Obeso I", "Obeso II+"],
    ).astype(str)

    # Score de estilo de vida saudável (0-100)
    activity_map = {"Sedentário": 0, "Leve": 25, "Moderado": 50, "Ativo": 75, "Muito Ativo": 100}
    df["lifestyle_score"] = (
        df["activity_level"].map(activity_map).fillna(50) * 0.4
        + df["adherence_pct"] * 0.4
        + (df["sleep_hours"].clip(6, 9) - 6) / 3 * 100 * 0.1
        + (df["water_intake_liters"].clip(1.5, 3) - 1.5) / 1.5 * 100 * 0.1
    ).round(2)

    # Peso por semana de dieta
    df["kg_per_week_expected"] = (df["weight_kg"] * 0.005).round(3)

    # Faixa etária
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 45, 55, 100],
        labels=["18-25", "26-35", "36-45", "46-55", "55+"],
    ).astype(str)

    # Interação: aderência × motivação
    df["adherence_x_motivation"] = (
        df["adherence_pct"] * df["initial_motivation"] / 100
    ).round(2)

    print(f"✅ Features engenheiradas adicionadas. Shape: {df.shape}")
    return df