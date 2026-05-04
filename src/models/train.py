"""
train.py
Treina modelos de classificação (diet_success) e regressão (weight_loss_kg).
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.pipeline import Pipeline

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TEST_SIZE   = 0.2
RANDOM_STATE = 42


# ── Classificação ──────────────────────────────────────────────────────────────
CLASSIFIERS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "DecisionTree":       DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "RandomForest":       RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "GradientBoosting":   GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
}

# ── Regressão ──────────────────────────────────────────────────────────────────
REGRESSORS = {
    "Ridge":              Ridge(),
    "DecisionTree":       DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE),
    "RandomForest":       RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
    "GradientBoosting":   GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE),
}


def train_classifiers(X: pd.DataFrame, y: pd.Series) -> dict:
    """Treina e retorna todos os classificadores + splits."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    trained = {}
    print("\n🔵 Treinando Classificadores...")
    for name, model in CLASSIFIERS.items():
        model.fit(X_train, y_train)
        cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring="f1").mean()
        trained[name] = {
            "model":   model,
            "cv_f1":   round(cv_score, 4),
            "X_test":  X_test,
            "y_test":  y_test,
        }
        print(f"   {name:<22} CV F1: {cv_score:.4f}")
        _save_model(model, f"classifier_{name}.pkl")

    return trained


def train_regressors(X: pd.DataFrame, y: pd.Series) -> dict:
    """Treina e retorna todos os regressores + splits."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    trained = {}
    print("\n🟢 Treinando Regressores...")
    for name, model in REGRESSORS.items():
        model.fit(X_train, y_train)
        cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error").mean()
        trained[name] = {
            "model":   model,
            "cv_rmse": round(abs(cv_score), 4),
            "X_test":  X_test,
            "y_test":  y_test,
        }
        print(f"   {name:<22} CV RMSE: {abs(cv_score):.4f}")
        _save_model(model, f"regressor_{name}.pkl")

    return trained


def _save_model(model, filename: str):
    path = MODELS_DIR / filename
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(filename: str):
    path = MODELS_DIR / filename
    with open(path, "rb") as f:
        return pickle.load(f)