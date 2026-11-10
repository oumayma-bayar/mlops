
# ============================================================
# 1. IMPORTER LES BIBLIOTHÈQUES
# ============================================================

import mlflow
import mlflow.sklearn
import pandas as pd


# ============================================================
# 2. RECHERCHER LE MEILLEUR RUN
# ============================================================

# Récupérer les runs de l'expérience churn-prediction
# Trier les résultats selon le score F1 décroissant

runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"]
)

# Récupérer le Run ID du meilleur modèle
run_id = runs.iloc[0]["run_id"]

print("Meilleur Run ID :", run_id)


# ============================================================
# 3. RECHARGER LE MEILLEUR MODÈLE
# ============================================================

# runs:/<run_id>/model désigne le modèle enregistré
# dans les artifacts du run sélectionné

model = mlflow.sklearn.load_model(
    f"runs:/{run_id}/model"
)

print("\n===== MODELE RECHARGE =====")
print("Type :", type(model).__name__)
print("Nombre de variables attendues :", model.n_features_in_)


# ============================================================
# 4. CRÉER UN NOUVEAU CLIENT
# ============================================================

# Exemple : client avec abonnement Standard
# Les abonnements sont déjà encodés en colonnes binaires

client = pd.DataFrame([{
    "age": 35,
    "salaire": 2500,
    "anciennete": 1,
    "consommation": 10,
    "abonnement_Premium": False,
    "abonnement_Standard": True
}])


# ============================================================
# 5. PRÉPARER LES DONNÉES POUR LE MODÈLE
# ============================================================

# Conserver uniquement les colonnes attendues par le modèle
# et respecter leur ordre

client = client[list(model.feature_names_in_)]


# ============================================================
# 6. EFFECTUER LA PRÉDICTION
# ============================================================

prediction = model.predict(client)[0]

print("\n===== RESULTAT DE LA PREDICTION =====")

if prediction == 1:
    print("Le client va partir : OUI")
else:
    print("Le client va partir : NON")
