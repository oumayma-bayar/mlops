
# ============================================================
# 1. IMPORTER LA BIBLIOTHÈQUE
# ============================================================

import requests


# ============================================================
# 2. DÉFINIR L'ADRESSE DE L'API
# ============================================================

URL = "http://localhost:8000"


# ============================================================
# 3. TESTER LA ROUTE GET /health
# ============================================================

# Vérifier que l'API est opérationnelle
response = requests.get(URL + "/health")

print("GET /health ->", response.status_code, response.json())


# ============================================================
# 4. TESTER LE PREMIER CLIENT
# ============================================================

# Client avec une faible ancienneté et une faible consommation
client1 = {
    "age": 35,
    "salaire": 2500,
    "abonnement": "Standard",
    "anciennete": 1,
    "consommation": 10
}

# Envoyer les données à la route /predict
response = requests.post(
    URL + "/predict",
    json=client1
)

print(
    "\nPOST /predict (client 1) ->",
    response.status_code,
    response.json()
)


# ============================================================
# 5. TESTER LE DEUXIÈME CLIENT
# ============================================================

# Client fidèle avec une forte ancienneté et consommation
client2 = {
    "age": 50,
    "salaire": 4000,
    "abonnement": "Premium",
    "anciennete": 10,
    "consommation": 80
}

response = requests.post(
    URL + "/predict",
    json=client2
)

print(
    "\nPOST /predict (client 2) ->",
    response.status_code,
    response.json()
)


# ============================================================
# 6. TESTER UNE ERREUR DE VALIDATION
# ============================================================

# Copier le premier client et remplacer son âge par du texte
mauvais = dict(client1, age="abc")

response = requests.post(
    URL + "/predict",
    json=mauvais
)

print(
    "\nPOST /predict (age='abc') ->",
    response.status_code
)

# Afficher le message d'erreur renvoyé par FastAPI
if response.status_code == 422:
    print("Message :", response.json()["detail"][0]["msg"])
else:
    print("Réponse :", response.json())
