
# ============================================================
# 1. IMPORTER LES BIBLIOTHÈQUES
# ============================================================

import mlflow
from mlflow import MlflowClient


# ============================================================
# 2. CONFIGURER LA CONNEXION À MLFLOW
# ============================================================

# Utiliser la base SQLite du TP3
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Créer un client pour gérer le Model Registry
client = MlflowClient()


# ============================================================
# 3. RETROUVER LE RUN "ESSAI-2"
# ============================================================

# Rechercher le run nommé "essai-2"
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    filter_string="tags.mlflow.runName = 'essai-2'"
)

# Vérifier que le run existe
if runs.empty:
    print("Aucun run nommé 'essai-2' trouvé.")

else:
    # Récupérer l'identifiant du run
    run_id = runs.iloc[0]["run_id"]

    print("Run trouvé : essai-2")
    print("ID :", run_id[:8], "...")


    # ========================================================
    # 4. ENREGISTRER UNE NOUVELLE VERSION DU MODÈLE
    # ========================================================

    # Enregistrer le modèle du run sous le nom "churn-model"
    v = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name="churn-model"
    )

    print("\nNouvelle version créée :", v.version)


    # ========================================================
    # 5. RÉCUPÉRER LE CHAMPION ACTUEL
    # ========================================================

    # Mémoriser la version actuellement désignée par l'alias
    actuel = client.get_model_version_by_alias(
        name="churn-model",
        alias="champion"
    )

    version_initiale = actuel.version

    print("Champion actuel : version", version_initiale)


    # ========================================================
    # 6. DÉPLACER L'ALIAS VERS LA NOUVELLE VERSION
    # ========================================================

    client.set_registered_model_alias(
        name="churn-model",
        alias="champion",
        version=v.version
    )

    print(
        "Après déplacement : champion -> version",
        client.get_model_version_by_alias(
            "churn-model", "champion"
        ).version
    )


    # ========================================================
    # 7. RÉTABLIR L'ALIAS SUR LA VERSION INITIALE
    # ========================================================

    client.set_registered_model_alias(
        name="churn-model",
        alias="champion",
        version=version_initiale
    )

    print(
        "Après retour : champion -> version",
        client.get_model_version_by_alias(
            "churn-model", "champion"
        ).version
    )
