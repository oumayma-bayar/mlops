
# ============================================================
# 1. IMPORTER LES BIBLIOTHÈQUES
# ============================================================

import mlflow
import mlflow.sklearn
import pandas as pd


# ============================================================
# 2. CONFIGURER LA CONNEXION À MLflow
# ============================================================

# Utiliser la même base SQLite que lors des TP précédents
mlflow.set_tracking_uri("sqlite:///mlflow.db")


# ============================================================
# 3. CHARGER LE MODÈLE CHAMPION
# ============================================================

# Charger le modèle à partir de son alias "champion"
# sans avoir besoin de connaître sa version
model = mlflow.sklearn.load_model(
    "models:/churn-model@champion"
)

print("Modèle chargé :", type(model).__name__)

# Afficher les variables attendues par le modèle
print("Variables attendues :", list(model.feature_names_in_))


# ============================================================
# 4. CRÉER UN NOUVEAU CLIENT
# ============================================================

# Exemple : client de 35 ans avec un abonnement Standard
# Les variables d'abonnement sont encodées en booléens (0/1)
client = pd.DataFrame([{
    "age": 35,
    "salaire": 2500,
    "anciennete": 1,
    "consommation": 10,
    "abonnement_Premium": False,
    "abonnement_Standard": True
}])


# ============================================================
# 5. PRÉPARER LES DONNÉES
# ============================================================

# Conserver uniquement les variables attendues par le modèle,
# dans le même ordre que pendant l'entraînement
client = client[list(model.feature_names_in_)]

print("\nDonnées du client :")
print(client)


# ============================================================
# 6. EFFECTUER LA PRÉDICTION
# ============================================================

# 1 = le client risque de partir
# 0 = le client reste
prediction = model.predict(client)[0]

if prediction == 1:
    resultat = "OUI"
else:
    resultat = "NON"

print("\nLe client va-t-il partir ? ->", resultat)
