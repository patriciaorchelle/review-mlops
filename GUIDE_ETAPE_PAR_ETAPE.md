# 🚀 Guide — Mini Projet MLOps à construire ce weekend

## Ce projet en une phrase
Un **classificateur d'avis produits** (POSITIF / NÉGATIF / SPAM) avec pipeline MLOps complet :
training tracké avec MLflow → API FastAPI → conteneurisé Docker → CI/CD GitHub Actions.

**Pourquoi ce projet ?**
- Directement lié au use case Decathlon (modération d'avis, lutte contre le spam)
- Couvre tout leur stack technique : Python, MLflow, Docker, FastAPI, GitHub Actions
- Faisable en **une après-midi**
- Tu pourras honnêtement dire en entretien : "J'ai construit un pipeline MLOps end-to-end"

---

## Prérequis (à installer si pas déjà fait)
- Python 3.10 ou 3.11
- Docker Desktop (https://www.docker.com/products/docker-desktop/)
- Git + compte GitHub
- VSCode ou tout autre éditeur

---

## ÉTAPE 1 — Setup du projet (15 min)

```bash
# Créer et activer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux/Mac
# OU
venv\Scripts\activate           # Windows

# Installer les dépendances
pip install -r requirements.txt

# Vérifier que tout est installé
python -c "import mlflow, fastapi, sklearn; print('OK')"
```

---

## ÉTAPE 2 — Créer les données + entraîner le modèle (30 min)

```bash
# Créer le dataset de reviews
python data/create_data.py
# → crée data/reviews.csv (60 avis : 20 positifs, 20 négatifs, 20 spam)

# Entraîner le modèle avec MLflow tracking
python src/train.py
# → entraîne TF-IDF + LogisticRegression
# → logge les métriques dans MLflow
# → sauvegarde models/pipeline.pkl

# Voir les expériences MLflow dans le navigateur
mlflow ui
# → ouvre http://localhost:5000
# → tu vois toutes les métriques, paramètres, et le modèle sauvegardé
```

**Ce que tu dois voir dans MLflow UI :**
- Un run avec les paramètres (n_features, C...)
- Les métriques (test_accuracy, test_f1, cv_f1_mean)
- Le modèle "review_classifier" dans les artefacts

**Maintenant teste plusieurs configurations :**
```bash
python src/train.py --n_features 5000 --C 0.1
python src/train.py --n_features 1000 --C 10.0
```
→ Compare les runs dans MLflow UI. C'est exactement ce qu'on fait en vrai !

---

## ÉTAPE 3 — Lancer l'API FastAPI (20 min)

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Tester l'API :**

Ouvrir http://localhost:8000/docs → interface interactive automatique (Swagger UI)

Ou avec curl :
```bash
# Health check
curl http://localhost:8000/health

# Prédiction
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Super produit, je recommande !"}'

# Prédiction batch
curl -X POST http://localhost:8000/predict/batch \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Très bon !", "Mauvaise qualité", "SPAM CLIQUEZ ICI"]}'
```

---

## ÉTAPE 4 — Lancer les tests (10 min)

```bash
pytest tests/ -v
```

Tu dois voir tous les tests passer en vert. Si un test échoue, lis le message d'erreur.

---

## ÉTAPE 5 — Conteneuriser avec Docker (30 min)

```bash
# Construire l'image Docker
docker build -t review-classifier:v1.0 .

# Vérifier que l'image est créée
docker images | grep review-classifier

# Lancer le container
docker run -p 8000:8000 review-classifier:v1.0

# Dans un autre terminal, tester
curl http://localhost:8000/health

# Arrêter le container
docker ps                     # trouver le container ID
docker stop <container_id>
```

**Important :**
- `docker build` → crée une IMAGE (blueprint)
- `docker run` → lance un CONTAINER à partir de l'image
- `-p 8000:8000` → mappe le port du container sur la machine

---

## ÉTAPE 6 — CI/CD avec GitHub Actions (30 min)

```bash
# Initialiser un repo git
git init
git add .
git commit -m "feat: initial MLOps pipeline"

# Créer un repo sur GitHub (github.com → New repository)
git remote add origin https://github.com/TON_USERNAME/decathlon-review-mlops.git
git push -u origin main
```

→ Va sur GitHub → onglet "Actions" → tu verras le pipeline CI/CD tourner automatiquement !
→ Il va : installer les dépendances → entraîner le modèle → lancer les tests → build Docker

---

## Ce que tu peux dire en entretien

**"Peux-tu me parler d'un projet MLOps ?"**

> "Oui, ce weekend j'ai construit un pipeline MLOps end-to-end sur un cas directement lié à votre use case : la classification d'avis produits en POSITIF, NÉGATIF et SPAM. Le pipeline complet inclut :
> - Un modèle TF-IDF + Logistic Regression entraîné et tracké avec MLflow — j'ai pu comparer plusieurs hyperparamètres dans l'UI et choisir le meilleur run
> - Une API FastAPI qui expose le modèle avec un endpoint /predict et un endpoint /predict/batch pour traiter plusieurs avis à la fois
> - Un Dockerfile qui conteneurise l'application avec un healthcheck pour Kubernetes
> - Un pipeline GitHub Actions qui lance les tests automatiquement à chaque push et build l'image Docker si tout passe
> Ce projet m'a permis de voir concrètement comment chaque brique s'emboîte, et je comprends maintenant pourquoi Docker + CI/CD sont indispensables pour garantir qu'un modèle se comporte de façon cohérente de la machine du data scientist jusqu'à la production."

---

## Sur Spark — Ce qu'il faut dire honnêtement

> "J'ai étudié Spark à Paris Cité et j'ai fait le cours de base. Pour ce projet de classification, le dataset était petit donc Pandas suffisait largement. Mais j'ai aussi écrit un exemple Spark pour montrer comment on ferait le même preprocessing à l'échelle de millions d'avis — tokenisation, TF-IDF distribué, agrégations par label. Je connais les concepts clés : RDD vs DataFrame, les transformations lazy vs les actions, et pourquoi Spark est indispensable quand Pandas ne tient plus en mémoire."

---

## Sur FastAPI — Ce qu'il faut dire

> "FastAPI est un framework Python très utilisé pour exposer des modèles ML. Sa force : il génère automatiquement la documentation Swagger, il valide les données d'entrée via Pydantic, et il est très performant (basé sur Starlette/async). J'ai utilisé FastAPI pour wrapper mon classificateur avec des endpoints /predict, /predict/batch et /health — ce dernier est indispensable pour que Kubernetes puisse vérifier l'état du service."
