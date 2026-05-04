"""
load_data.py
Carrega os 4 CSVs brutos e retorna DataFrames com validação básica.
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

EXPECTED_FILES = {
    "patients":       "patients.csv",
    "diets":          "diets.csv",
    "nutritionists":  "nutritionists.csv",
    "results":        "results.csv",
}


def load_all(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    """
    Retorna um dicionário com os 4 DataFrames brutos.
    Exemplo:
        data = load_all()
        df_patients = data["patients"]
    """
    data = {}
    for key, filename in EXPECTED_FILES.items():
        filepath = raw_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {filepath}\n"
                f"Execute primeiro: python src/data/generate_synthetic_data.py"
            )
        df = pd.read_csv(filepath)
        print(f"✅ {filename}: {df.shape[0]} linhas × {df.shape[1]} colunas")
        data[key] = df
    return data


def quick_info(data: dict[str, pd.DataFrame]) -> None:
    """Imprime um resumo rápido de cada dataset."""
    for name, df in data.items():
        print(f"\n{'─'*40}")
        print(f"📋 {name.upper()}")
        print(df.dtypes.to_string())
        print(f"Nulos: {df.isnull().sum().sum()}")