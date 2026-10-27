
# ============================================================
# 1. IMPORTER LES BIBLIOTHÈQUES
# ============================================================

import mlflow
from mlflow import MlflowClient


# ============================================================
# 2. CONFIGURER LA CONNEXION À LA BASE MLflow
# ============================================================

# Utiliser la base SQLite du TP3
mlflow.set_tracking_uri("sqlite:///mlflow.db")


# ============================================================
# 3. RÉCUPÉRER LE MEILLEUR RUN
# ============================================================

# Récupérer les runs de l'expérience et les trier
# par F1-score décroissant
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"]
)

# Vérifier qu'au moins un run existe
if runs.empty:
    print("Aucun run trouvé dans l'expérience churn-prediction.")

else:
    # Sélectionner le meilleur run
    best = runs.iloc[0]

    run_id = best["run_id"]
    run_name = best["tags.mlflow.runName"]

    print("Meilleur run :", run_name)
    print("ID :", run_id[:8], "...")
    print("F1-score :", round(best["metrics.f1"], 3))


    # ========================================================
    # 4. ENREGISTRER LE MODÈLE DANS LE MODEL REGISTRY
    # ========================================================

    # Enregistrer le modèle du meilleur run
    # sous le nom "churn-model"
    result = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name="churn-model"
    )

    print(
        "\nModèle enregistré : churn-model",
        "| Version :", result.version
    )


    # ========================================================
    # 5. ATTRIBUER L'ALIAS "champion"
    # ========================================================

    # Créer un client MLflow
    client = MlflowClient()

    # Désigner cette version comme modèle champion
    client.set_registered_model_alias(
        name="churn-model",
        alias="champion",
        version=result.version
    )

    print(
        "Alias 'champion' -> version",
        result.version
    )
