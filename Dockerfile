
# ==========================================
# 1. Image de base : Python 3.11
# ==========================================
FROM python:3.11-slim

# ==========================================
# 2. Définir le dossier de travail
# ==========================================
WORKDIR /app

# ==========================================
# 3. Installer les dépendances
# ==========================================
# Copier requirements.txt en premier pour profiter du cache Docker
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ==========================================
# 4. Copier le code source et le modèle
# ==========================================
COPY app.py .
COPY model/ model/

# ==========================================
# 5. Configurer la variable d'environnement
# ==========================================
ENV MODEL_URI=model

# ==========================================
# 6. Exposer le port de l'API
# ==========================================
EXPOSE 8000

# ==========================================
# 7. Démarrer l'application FastAPI
# ==========================================
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
