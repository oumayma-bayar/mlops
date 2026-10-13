
# ============================================================
# 1. Importation des bibliothèques
# ============================================================

import pandas as pd
import numpy as np


# ============================================================
# 2. Initialisation et paramètres
# ============================================================

# Fixer la graine aléatoire pour garantir la reproductibilité
np.random.seed(42)

# Nombre de clients fictifs à générer
n = 500


# ============================================================
# 3. Génération des données des clients (FEATURES)
# ============================================================

data = pd.DataFrame({
    "age": np.random.randint(18, 70, n),
    "salaire": np.random.randint(1200, 8000, n),
    "abonnement": np.random.choice(
        ["Basique", "Standard", "Premium"], n
    ),
    "anciennete": np.random.randint(0, 15, n),
    "consommation": np.random.randint(0, 100, n)
})


# ============================================================
# 4. Création de la variable cible (TARGET)
# ============================================================

# churn = 1 si la consommation < 30 ET l'ancienneté < 2 ans
# churn = 0 dans les autres cas

data["churn"] = (
    (data["consommation"] < 30) &
    (data["anciennete"] < 2)
).astype(int)


# ============================================================
# 5. Sauvegarde du dataset au format CSV
# ============================================================

data.to_csv("churn_data.csv", index=False)


# ============================================================
# 6. Affichage et vérification des résultats
# ============================================================

# Afficher les cinq premières lignes
print("Aperçu du dataset :")
print(data.head())

# Afficher les dimensions du dataset
print("\nDimensions du dataset :")
print(data.shape)

# Afficher le nombre de clients par classe
print("\nRépartition du churn :")
print(data["churn"].value_counts())

# Afficher le taux de churn
print(f"\nTaux de churn : {data['churn'].mean():.1%}")

# Confirmer la sauvegarde
print("\nDataset enregistré dans churn_data.csv")
