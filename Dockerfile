# ─────────────────────────────────────────────────────────
# Dockerfile — Decathlon Review Classifier
# ─────────────────────────────────────────────────────────
# Étape 1 : image de base Python légère
FROM python:3.11-slim

# Qui maintient cette image (bonne pratique)
LABEL maintainer="Patricia Welehela <patriciawelehela@gmail.com>"
LABEL description="API de classification d'avis"
LABEL version="1.0.0"

# ─── VARIABLES D'ENVIRONNEMENT ────────────────────────────
# Empêche Python de créer des fichiers .pyc inutiles
ENV PYTHONDONTWRITEBYTECODE=1
# Force les logs à apparaître en temps réel (pas de buffer)
ENV PYTHONUNBUFFERED=1
# Port exposé par l'API
ENV PORT=8000

# ─── RÉPERTOIRE DE TRAVAIL ────────────────────────────────
WORKDIR /app

# ─── DÉPENDANCES ─────────────────────────────────────────
# Copier d'abord seulement requirements.txt
# (pour profiter du cache Docker si le code change mais pas les dépendances)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── CODE SOURCE ─────────────────────────────────────────
# Copier tout le reste
COPY . .

# Créer le répertoire models (il sera rempli par le volume ou à la build)
RUN mkdir -p models

# ─── PORT EXPOSÉ ─────────────────────────────────────────
EXPOSE 8000

# ─── HEALTHCHECK ─────────────────────────────────────────
# Kubernetes utilise ça pour vérifier que le container est sain
#Quand le container tourne, Docker exécute cette commande toutes les 30 secondes pour vérifier que l'API est vivante.

#--interval=30s → vérifie toutes les 30 secondes
#--start-period=5s → attend 5 secondes au démarrage avant de commencer à vérifier (le temps que uvicorn démarre)
#--timeout=10s → si pas de réponse en 10 secondes → échec
#--retries=3 → 3 échecs consécutifs → container marqué "unhealthy"

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# ─── COMMANDE DE DÉMARRAGE ────────────────────────────────
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
