
# ============================================================
# 1. IMPORTER MLFLOW
# ============================================================

import mlflow


# ============================================================
# 2. RÉCUPÉRER ET TRIER LES RUNS
# ============================================================

# Récupérer tous les essais de l'expérience
# et les trier par F1-score décroissant
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"],
)


# ============================================================
# 3. AFFICHER LES RÉSULTATS
# ============================================================

# Colonnes à afficher
colonnes = [
    "tags.mlflow.runName",
    "params.n_estimators",
    "params.max_depth",
    "metrics.accuracy",
    "metrics.f1",
]

# Vérifier si des essais existent
if runs.empty:
    print("Aucun run trouvé dans l'expérience churn-prediction.")

else:
    # Afficher le tableau sans les numéros de ligne
    print("Comparaison des essais MLflow :\n")
    print(runs[colonnes].to_string(index=False))

    # ========================================================
    # 4. IDENTIFIER LE MEILLEUR RUN
    # ========================================================

    # Le premier run possède le F1-score le plus élevé
    meilleur = runs.iloc[0]

    print("\nMeilleur essai :")
    print(
        "Nom :", meilleur["tags.mlflow.runName"],
        "| ID :", meilleur["run_id"][:8] + "...",
    )
    print("Accuracy :", round(meilleur["metrics.accuracy"], 3))
    print("F1-score :", round(meilleur["metrics.f1"], 3))
