"""
plots.py
Funções de visualização para EDA e resultados de modelos.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")


def _save(fig, name: str):
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Figura salva: {path.name}")


# ── EDA ────────────────────────────────────────────────────────────────────────
def plot_target_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Classificação
    counts = df["diet_success"].value_counts()
    axes[0].bar(["Insucesso", "Sucesso"], counts.values, color=["#e74c3c", "#2ecc71"])
    axes[0].set_title("Distribuição: Sucesso da Dieta")
    axes[0].set_ylabel("Quantidade")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 2, str(v), ha="center", fontweight="bold")

    # Regressão
    axes[1].hist(df["weight_loss_kg"], bins=30, color="#3498db", edgecolor="white")
    axes[1].set_title("Distribuição: Perda de Peso (kg)")
    axes[1].set_xlabel("kg perdidos")
    axes[1].set_ylabel("Frequência")

    _save(fig, "target_distribution.png")


def plot_correlation_heatmap(df: pd.DataFrame):
    num_df = df.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(num_df.corr(), dtype=bool))
    sns.heatmap(num_df.corr(), mask=mask, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, ax=ax, linewidths=0.5)
    ax.set_title("Mapa de Correlação")
    _save(fig, "correlation_heatmap.png")


def plot_diet_vs_success(df: pd.DataFrame, diets_df: pd.DataFrame):
    """Taxa de sucesso por tipo de dieta."""
    merged = df.merge(diets_df[["diet_id", "diet_name"]], on="diet_id", how="left")
    success_rate = merged.groupby("diet_name")["diet_success"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(success_rate.index, success_rate.values * 100, color="#2980b9")
    ax.set_xlabel("Taxa de Sucesso (%)")
    ax.set_title("Taxa de Sucesso por Tipo de Dieta")
    for bar, val in zip(bars, success_rate.values):
        ax.text(val * 100 + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val*100:.1f}%", va="center")
    _save(fig, "diet_vs_success.png")


def plot_age_vs_weight_loss(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["age"], df["weight_loss_kg"], alpha=0.4, color="#8e44ad", s=20)
    m, b = np.polyfit(df["age"], df["weight_loss_kg"], 1)
    ax.plot(df["age"].sort_values(), m * df["age"].sort_values() + b,
            color="red", linewidth=1.5, label=f"Tendência: y={m:.2f}x+{b:.2f}")
    ax.set_xlabel("Idade")
    ax.set_ylabel("Perda de Peso (kg)")
    ax.set_title("Idade vs Perda de Peso")
    ax.legend()
    _save(fig, "age_vs_weight_loss.png")


def plot_adherence_vs_success(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color, name in [(0, "#e74c3c", "Insucesso"), (1, "#2ecc71", "Sucesso")]:
        subset = df[df["diet_success"] == label]["adherence_pct"]
        ax.hist(subset, bins=20, alpha=0.6, color=color, label=name)
    ax.set_xlabel("Aderência (%)")
    ax.set_ylabel("Frequência")
    ax.set_title("Aderência à Dieta vs Sucesso")
    ax.legend()
    _save(fig, "adherence_vs_success.png")


def plot_feature_importance(importance_df: pd.DataFrame, title: str = "Feature Importance"):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=importance_df, x="Importance", y="Feature", ax=ax, palette="Blues_r")
    ax.set_title(title)
    _save(fig, f"feature_importance_{title.lower().replace(' ', '_')}.png")


def plot_model_comparison(results_df: pd.DataFrame, metric: str, title: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("muted", len(results_df))
    bars = ax.bar(results_df["Model"], results_df[metric], color=colors)
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_ylim(0, results_df[metric].max() * 1.15)
    for bar, val in zip(bars, results_df[metric]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", fontsize=9)
    plt.xticks(rotation=15)
    _save(fig, f"model_comparison_{metric.lower()}.png")