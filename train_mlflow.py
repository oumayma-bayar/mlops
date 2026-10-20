
# ============================================================
# 1. IMPORTER LES BIBLIOTHÈQUES
# ============================================================

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# 2. CHARGER LES DONNÉES
# ============================================================

# Charger le dataset contenant les 500 clients
df = pd.read_csv("churn_data.csv")

# Variable cible :
# 1 = le client part
# 0 = le client reste
y = df["churn"]


# ============================================================
# 3. FONCTION DE PRÉPARATION DES DONNÉES
# ============================================================

def preparer(colonnes):

    # Sélectionner les colonnes souhaitées
    X = df[colonnes]

    # Convertir les variables catégorielles en colonnes numériques
    X = pd.get_dummies(X, drop_first=True)

    # Séparer les données en apprentissage (80 %) et test (20 %)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


# ============================================================
# 4. FONCTION POUR LANCER ET ENREGISTRER UN ESSAI
# ============================================================

def lancer_essai(nom, n_estimators, max_depth, colonnes):

    # Préparer les données
    X_train, X_test, y_train, y_test = preparer(colonnes)

    # Ouvrir un run MLflow
    with mlflow.start_run(run_name=nom):

        # Créer le modèle Random Forest
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )

        # Entraîner le modèle
        model.fit(X_train, y_train)

        # Effectuer les prédictions
        pred = model.predict(X_test)

        # Calculer les métriques
        accuracy = accuracy_score(y_test, pred)

        precision = precision_score(
            y_test, pred, zero_division=0
        )

        recall = recall_score(
            y_test, pred, zero_division=0
        )

        f1 = f1_score(
            y_test, pred, zero_division=0
        )

        # ----------------------------------------------------
        # Enregistrer les paramètres dans MLflow
        # ----------------------------------------------------

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("features", ",".join(colonnes))

        # ----------------------------------------------------
        # Enregistrer les métriques dans MLflow
        # ----------------------------------------------------

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        # ----------------------------------------------------
        # Enregistrer le modèle comme artefact
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format="cloudpickle",
        )

        # Afficher les résultats
        print(
            f"{nom} -> "
            f"Accuracy : {accuracy:.3f} | "
            f"Precision : {precision:.3f} | "
            f"Recall : {recall:.3f} | "
            f"F1 : {f1:.3f}"
        )


# ============================================================
# 5. PROGRAMME PRINCIPAL
# ============================================================

# Définir l'expérience MLflow
mlflow.set_experiment("churn-prediction")

# Toutes les variables explicatives
TOUT = [
    "age",
    "salaire",
    "abonnement",
    "anciennete",
    "consommation",
]

# ------------------------------------------------------------
# ESSAI 1 : 100 arbres, profondeur maximale de 5
# Toutes les variables
# ------------------------------------------------------------

lancer_essai(
    "essai-1",
    100,
    5,
    TOUT,
)

# ------------------------------------------------------------
# ESSAI 2 : 50 arbres, profondeur maximale de 3
# Toutes les variables
# ------------------------------------------------------------

lancer_essai(
    "essai-2",
    50,
    3,
    TOUT,
)

# ------------------------------------------------------------
# ESSAI 3 : retirer la variable salaire
# 100 arbres, profondeur maximale de 5
# ------------------------------------------------------------

lancer_essai(
    "essai-3",
    100,
    5,
    [
        "age",
        "abonnement",
        "anciennete",
        "consommation",
    ],
)
