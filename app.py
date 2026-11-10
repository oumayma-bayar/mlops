
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
import joblib
import os

# --------------------------------------------------
# 1. Initialisation de FastAPI
# --------------------------------------------------

app = FastAPI(
    title="API Churn Prediction",
    description="API pour prédire si un client risque de quitter l'entreprise",
    version="1.0.0"
)

# --------------------------------------------------
# 2. Chargement du modèle
# --------------------------------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model",
    "model.pkl"
)

try:
    model = joblib.load(MODEL_PATH)
    print("Modèle chargé avec succès !")
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du modèle : {e}")


# --------------------------------------------------
# 3. Schéma des données d'entrée
# --------------------------------------------------

class ClientData(BaseModel):
    age: int = Field(..., ge=18, le=100)
    abonnement: Literal["Basique", "Standard", "Premium"]
    anciennete: float = Field(..., ge=0)
    consommation: float = Field(..., ge=0)


# --------------------------------------------------
# 4. Route d'accueil
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Bienvenue dans l'API Churn Prediction",
        "documentation": "/docs"
    }


# --------------------------------------------------
# 5. Vérification de l'état de l'API
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "OK",
        "model_loaded": model is not None
    }


# --------------------------------------------------
# 6. Préparation des données
# --------------------------------------------------

def prepare_data(client: ClientData):

    # Convertir les données en DataFrame
    data = pd.DataFrame([{
        "age": client.age,
        "abonnement": client.abonnement,
        "anciennete": client.anciennete,
        "consommation": client.consommation
    }])

    # IMPORTANT :
    # Le modèle attend exactement les colonnes utilisées
    # pendant son entraînement.
    expected_columns = list(model.feature_names_in_)

    # Encodage de l'abonnement
    # À adapter si l'entraînement utilisait un autre encodage.
    abonnement_mapping = {
        "Basique": 0,
        "Standard": 1,
        "Premium": 2
    }

    data["abonnement"] = data["abonnement"].map(abonnement_mapping)

    # Réorganiser les colonnes selon le modèle
    data = data[expected_columns]

    return data


# --------------------------------------------------
# 7. Prédiction du churn
# --------------------------------------------------

@app.post("/predict")
def predict(client: ClientData):

    try:
        # Préparer les données
        data = prepare_data(client)

        # Faire la prédiction
        prediction = model.predict(data)[0]

        # Probabilité si le modèle la prend en charge
        probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(data)[0]
            probability = float(probabilities[1])

        # Résultat
        resultat = "Client susceptible de quitter" if int(prediction) == 1 else "Client fidèle"

        return {
            "prediction": int(prediction),
            "resultat": resultat,
            "probabilite_churn": probability
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur pendant la prédiction : {str(e)}"
        )
