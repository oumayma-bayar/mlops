
# ============================================================
# 1. IMPORTER LES BIBLIOTHÈQUES
# ============================================================

from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# ============================================================
# 2. CRÉER L'APPLICATION FASTAPI
# ============================================================

app = FastAPI(
    title="API Churn",
    description="API de prédiction du départ des clients",
    version="1.0.0"
)


# ============================================================
# 3. CONFIGURER MLFLOW ET CHARGER LE MODÈLE CHAMPION
# ============================================================

# Utiliser la base SQLite du projet
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Charger le modèle une seule fois au démarrage
model = mlflow.sklearn.load_model(
    "models:/churn-model@champion"
)

print("Modèle chargé :", type(model).__name__)


# ============================================================
# 4. DÉFINIR LE FORMAT DES DONNÉES DU CLIENT
# ============================================================

class Client(BaseModel):
    age: int

    salaire: float

    abonnement: Literal[
        "Basique",
        "Standard",
        "Premium"
    ]

    anciennete: float

    consommation: float


# ============================================================
# 5. ROUTE GET /health : VÉRIFIER L'ÉTAT DE L'API
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# ============================================================
# 6. ROUTE POST /predict : PRÉDIRE LE CHURN
# ============================================================

@app.post("/predict")
def predict(client: Client):

    # --------------------------------------------------------
    # a) Convertir les données reçues en DataFrame
    # --------------------------------------------------------

    ligne = pd.DataFrame([
        client.model_dump()
    ])


    # --------------------------------------------------------
    # b) Encoder la variable catégorielle abonnement
    # --------------------------------------------------------

    ligne = pd.get_dummies(ligne)


    # --------------------------------------------------------
    # c) Adapter les colonnes aux attentes du modèle
    # --------------------------------------------------------

    ligne = ligne.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )


    # --------------------------------------------------------
    # d) Effectuer la prédiction
    # --------------------------------------------------------

    resultat = int(
        model.predict(ligne)[0]
    )


    # --------------------------------------------------------
    # e) Retourner le résultat au format JSON
    # --------------------------------------------------------

    return {
        "churn": resultat,
        "message": (
            "Le client va partir"
            if resultat == 1
            else "Le client reste"
        )
    }
