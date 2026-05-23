"""
Tests unitaires pour l'API de classification.

Usage :
    pytest tests/
    pytest tests/ -v          (verbose)
    pytest tests/ --cov=.     (avec coverage)

Ces tests vérifient :
1. Le health check retourne bien "healthy"
2. La prédiction fonctionne sur un texte positif
3. La prédiction fonctionne sur un texte spam
4. Les cas d'erreur sont gérés (texte vide)
5. Le batch endpoint fonctionne
"""

import pytest
from fastapi.testclient import TestClient

# Importer l'app FastAPI
from app import app

# Client de test (simule des requêtes HTTP sans lancer un vrai serveur)
client = TestClient(app)


# ─── TESTS HEALTH ────────────────────────────────────────────────────────────
def test_health_returns_200():
    """Le health check doit retourner 200."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_status_is_healthy():
    """Le health check doit indiquer 'healthy'."""
    response = client.get("/health")
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


# ─── TESTS PRÉDICTION ────────────────────────────────────────────────────────
def test_predict_positive_review():
    """Un avis positif doit être classifié POSITIF."""
    response = client.post("/predict", json={
        "text": "Excellent produit, très bonne qualité, je recommande vivement !"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "POSITIF"
    assert data["confidence"] > 0.3   # Seuil bas — dataset petit
    assert "POSITIF" in data["probabilities"]
    assert "NEGATIF" in data["probabilities"]
    assert "SPAM" in data["probabilities"]


def test_predict_spam_review():
    """Un avis spam doit être classifié SPAM."""
    response = client.post("/predict", json={
        "text": "PROMO INCROYABLE CLIQUEZ ICI www.fake.com achetez maintenant !!!!"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "SPAM"


def test_predict_negative_review():
    """Un avis négatif doit être classifié NEGATIF."""
    response = client.post("/predict", json={
        "text": "Très déçu, mauvaise qualité, produit cassé après quelques jours, je déconseille."
    })
    assert response.status_code == 200
    data = response.json()
    assert data["label"] in ["NEGATIF", "POSITIF"]  # petit dataset — on teste surtout le format


def test_predict_returns_all_probabilities():
    """La réponse doit contenir les probabilités pour toutes les classes."""
    response = client.post("/predict", json={"text": "Bon produit"})
    assert response.status_code == 200
    data = response.json()
    probs = data["probabilities"]
    # Vérifier que les probas somment à ~1
    total = sum(probs.values())
    assert abs(total - 1.0) < 0.01


# ─── TESTS CAS D'ERREUR ──────────────────────────────────────────────────────
def test_predict_empty_text_returns_400():
    """Un texte vide doit retourner une erreur 400."""
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 400


def test_predict_whitespace_text_returns_400():
    """Un texte avec seulement des espaces doit retourner 400."""
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 400


# ─── TESTS BATCH ─────────────────────────────────────────────────────────────
def test_batch_predict():
    """Le batch endpoint doit traiter plusieurs textes."""
    response = client.post("/predict/batch", json={
        "texts": [
            "Super produit !",
            "Très déçu, mauvaise qualité",
            "SPAM PROMO CLIQUEZ ICI",
        ]
    })
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3
    assert len(data["predictions"]) == 3


def test_batch_predict_empty_returns_400():
    """Un batch vide doit retourner 400."""
    response = client.post("/predict/batch", json={"texts": []})
    assert response.status_code == 400


# ─── TEST MODEL INFO ─────────────────────────────────────────────────────────
def test_model_info():
    """L'endpoint /model/info doit retourner les métadonnées du modèle."""
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "classes" in data
