
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Connexion à MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = MlflowClient()

# Rechercher le meilleur Run
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"]
)

if runs.empty:
    raise ValueError("Aucun Run trouvé.")

run_id = runs.iloc[0]["run_id"]

print("Meilleur Run ID :", run_id)
print("Score F1 :", runs.iloc[0]["metrics.f1"])

# Vérifier les artefacts
artifacts = client.list_artifacts(run_id)

print("\nArtefacts disponibles :")

for artifact in artifacts:
    print(artifact.path)

# Charger le modèle
model = mlflow.sklearn.load_model(
    f"runs:/{run_id}/model"
)

# Exporter dans model/
mlflow.sklearn.save_model(
    model,
    path="model",
    serialization_format="cloudpickle"
)

print("\nChampion exporté avec succès dans model/")
