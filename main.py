"""
main.py
Pipeline principal do NutriPredict AI
"""

from src.data.load_data import load_all
from src.data.merge_data import merge_all
from src.features.feature_engineering import add_features
from src.data.preprocess import preprocess

from src.models.train import train_classifiers, train_regressors
from src.models.evaluate import evaluate_classifiers, evaluate_regressors, feature_importance

from src.features.feature_selection import run_feature_selection

from src.clustering.clustering import (
    find_optimal_k,
    run_kmeans,
    run_hierarchical,
    plot_clusters_pca,
    cluster_profile
)

from src.visualization.plots import (
    plot_target_distribution,
    plot_correlation_heatmap,
    plot_correlation_with_target,
    plot_gender_vs_success,
    plot_diet_vs_success,
    plot_age_vs_weight_loss,
    plot_age_group_vs_success,
    plot_adherence_vs_success,
    plot_motivation_vs_success,
    plot_bmi_vs_weight_loss,
    plot_outliers_boxplot,
    plot_clustering_comparison,
    plot_nutritionist_vs_outcome,
    plot_specialty_vs_success,
    plot_model_comparison,
    plot_feature_importance,
)


def main():
    print("=" * 60)
    print("NUTRIPREDICT AI - PIPELINE")
    print("=" * 60)

    # --------------------------------------------------
    # 1. LOAD
    # --------------------------------------------------
    print("\n[1] Carregando dados...")
    data = load_all()

    # --------------------------------------------------
    # 2. MERGE
    # --------------------------------------------------
    print("\n[2] Integrando datasets...")
    df = merge_all(data)

    # --------------------------------------------------
    # 3. FEATURE ENGINEERING
    # --------------------------------------------------
    print("\n[3] Criando features...")
    df = add_features(df)

    # --------------------------------------------------
    # 4. EDA
    # --------------------------------------------------
    print("\n[4] Gerando análises exploratórias...")

    # Distribuição dos targets
    plot_target_distribution(df)

    # Correlações
    plot_correlation_heatmap(df)
    plot_correlation_with_target(df)

    # Sexo
    plot_gender_vs_success(df)

    # Dieta
    plot_diet_vs_success(df)

    # Idade
    plot_age_vs_weight_loss(df)
    plot_age_group_vs_success(df)

    # Aderência e motivação
    plot_adherence_vs_success(df)
    plot_motivation_vs_success(df)

    # Outliers (antes do preprocess)
    plot_outliers_boxplot(df)

    # IMC
    plot_bmi_vs_weight_loss(df)

    # Nutricionista
    plot_nutritionist_vs_outcome(df)
    plot_specialty_vs_success(df)

    # --------------------------------------------------
    # 5. PREPROCESS
    # --------------------------------------------------
    print("\n[5] Pré-processando...")
    processed = preprocess(df)

    X       = processed["X"]
    y_class = processed["y_class"]
    y_reg   = processed["y_reg"]

    # --------------------------------------------------
    # 6. SELEÇÃO DE FEATURES
    # --------------------------------------------------
    print("\n[6] Seleção de features...")
    selected_features = run_feature_selection(X, y_class, y_reg)

    # --------------------------------------------------
    # 7. CLUSTERING
    # --------------------------------------------------
    print("\n[6] Executando clustering...")
    X_clustering = X.select_dtypes(include=["number"])

    k      = find_optimal_k(X_clustering)
    labels = run_kmeans(X_clustering, k)
    plot_clusters_pca(X_clustering, labels)
    cluster_profile(X, labels)

    # --------------------------------------------------
    # 7. TREINO
    # --------------------------------------------------
    print("\n[8] Treinando classificadores...")
    classifiers = train_classifiers(X, y_class)

    print("\n[9] Treinando regressores...")
    regressors = train_regressors(X, y_reg)

    # --------------------------------------------------
    # 9. AVALIAÇÃO
    # --------------------------------------------------
    print("\n[10] Avaliando modelos...")
    class_results = evaluate_classifiers(classifiers)
    reg_results   = evaluate_regressors(regressors)

    # --------------------------------------------------
    # 9. GRÁFICOS DE COMPARAÇÃO
    # --------------------------------------------------
    plot_model_comparison(class_results, "F1_teste",  "Comparação de Classificadores")
    plot_model_comparison(reg_results,   "R2_teste",  "Comparação de Regressores")

    # --------------------------------------------------
    # 10. FEATURE IMPORTANCE
    # --------------------------------------------------
    best_rf    = classifiers["RandomForest"]["model"]
    importance = feature_importance(best_rf, list(X.columns))

    if not importance.empty:
        plot_feature_importance(importance, "Random Forest")

    print("\n" + "=" * 60)
    print("PIPELINE FINALIZADO COM SUCESSO")
    print("=" * 60)


if __name__ == "__main__":
    main()