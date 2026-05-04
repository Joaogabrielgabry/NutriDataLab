"""
preprocess.py
Limpeza, encoding, normalização e tratamento de outliers.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Colunas que não entram no modelo
DROP_COLS = ["patient_id", "diet_id", "nutritionist_id"]

# Colunas categóricas para encoding
CATEGORICAL_COLS = [
    "gender", "activity_level", "health_condition",
    "specialty", "difficulty_level",
]

# Colunas numéricas para normalização
SCALE_COLS = [
    "age", "height_cm", "weight_kg", "bmi", "diet_duration_weeks",
    "adherence_pct", "initial_motivation", "sleep_hours",
    "water_intake_liters", "experience_years", "avg_patient_rating",
    "avg_daily_calories", "carb_pct", "protein_pct", "fat_pct",
]


def remove_outliers_iqr(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Remove linhas com outliers extremos (além de 3×IQR)."""
    mask = pd.Series(True, index=df.index)
    for col in cols:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr     = q3 - q1
        lower   = q1 - 3 * iqr
        upper   = q3 + 3 * iqr
        mask &= df[col].between(lower, upper)
    removed = (~mask).sum()
    if removed:
        print(f"⚠️  Outliers removidos: {removed} linhas")
    return df[mask].reset_index(drop=True)


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Aplica encoding em TODAS as colunas categóricas automaticamente.
    """
    encoders = {}

    cat_cols = df.select_dtypes(include="object").columns

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


def scale_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Aplica StandardScaler nas colunas numéricas.
    Retorna (df_scaled, scaler).
    """
    cols_present = [c for c in SCALE_COLS if c in df.columns]
    scaler = StandardScaler()
    df[cols_present] = scaler.fit_transform(df[cols_present])
    return df, scaler


def preprocess(df_raw: pd.DataFrame, scale: bool = True) -> dict:
    """
    Pipeline completo de pré-processamento.

    Parâmetros
    ----------
    df_raw : DataFrame consolidado (saída do merge_all)
    scale  : se True, aplica StandardScaler

    Retorna
    -------
    dict com:
        - df        : DataFrame pronto para modelagem
        - encoders  : dict de LabelEncoders por coluna
        - scaler    : StandardScaler (ou None)
        - X         : features (sem target)
        - y_class   : target de classificação (diet_success)
        - y_reg     : target de regressão (weight_loss_kg)
    """
    df = df_raw.copy()

    # 1. Remove colunas de ID
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # 2. Trata nulos: numéricos → mediana, categóricos → moda
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # 3. Remove outliers
    num_cols = df.select_dtypes(include="number").columns.tolist()
    df = remove_outliers_iqr(df, num_cols)

    # 4. Separa targets antes de escalar
    y_class = df.pop("diet_success")
    y_reg   = df.pop("weight_loss_kg")

    # 5. Remove colunas muito correlacionadas ao target para evitar leakage
    leakage_cols = ["final_bmi", "satisfaction_score"]
    df.drop(columns=[c for c in leakage_cols if c in df.columns], inplace=True)

    # 6. Encoding
    df, encoders = encode_categoricals(df)

    # 7. Normalização
    scaler = None
    if scale:
        df, scaler = scale_features(df)

    # Salva
    df_out = df.copy()
    df_out["diet_success"]   = y_class.values
    df_out["weight_loss_kg"] = y_reg.values
    df_out.to_csv(PROCESSED_DIR / "dataset_processed.csv", index=False)
    print(f"✅ Dados processados salvos em: {PROCESSED_DIR / 'dataset_processed.csv'}")

    return {
        "df":       df,
        "encoders": encoders,
        "scaler":   scaler,
        "X":        df,
        "y_class":  y_class,
        "y_reg":    y_reg,
    }