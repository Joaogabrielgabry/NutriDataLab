"""
evaluate.py
Avalia e compara modelos de classificação e regressão.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
)


# ── Classificação ──────────────────────────────────────────────────────────────
def evaluate_classifiers(trained_models: dict) -> pd.DataFrame:
    """
    Avalia todos os classificadores no conjunto de teste.
    Retorna um DataFrame com as métricas comparativas.
    """
    rows = []
    print("\n📊 Avaliação — Classificação (diet_success)\n" + "─" * 55)
    for name, info in trained_models.items():
        model, X_test, y_test = info["model"], info["X_test"], info["y_test"]
        y_pred = model.predict(X_test)
        row = {
            "Model":     name,
            "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
            "CV_F1":     info.get("cv_f1", None),
        }
        rows.append(row)
        print(f"  {name}")
        print(classification_report(y_test, y_pred, target_names=["Insucesso", "Sucesso"]))
    results = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    print("\n🏆 Ranking Classificação:")
    print(results.to_string(index=False))
    return results


def evaluate_regressors(trained_models: dict) -> pd.DataFrame:
    """
    Avalia todos os regressores no conjunto de teste.
    Retorna um DataFrame com as métricas comparativas.
    """
    rows = []
    print("\n📊 Avaliação — Regressão (weight_loss_kg)\n" + "─" * 55)
    for name, info in trained_models.items():
        model, X_test, y_test = info["model"], info["X_test"], info["y_test"]
        y_pred = model.predict(X_test)
        rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
        row = {
            "Model":    name,
            "RMSE":     round(rmse, 4),
            "MAE":      round(mean_absolute_error(y_test, y_pred), 4),
            "R2":       round(r2_score(y_test, y_pred), 4),
            "CV_RMSE":  info.get("cv_rmse", None),
        }
        rows.append(row)
        print(f"  {name:<22} RMSE={rmse:.3f} | MAE={row['MAE']:.3f} | R²={row['R2']:.3f}")
    results = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    print("\n🏆 Ranking Regressão:")
    print(results.to_string(index=False))
    return results


def feature_importance(model, feature_names: list, top_n: int = 10) -> pd.DataFrame:
    """Extrai importância das features (para tree-based models)."""
    if not hasattr(model, "feature_importances_"):
        print("⚠️  Modelo não suporta feature_importances_")
        return pd.DataFrame()
    imp = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).head(top_n)
    return imp