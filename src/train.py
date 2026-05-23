"""
Pipeline d'entraînement avec MLflow tracking.

Ce script entraîne un classificateur de reviews produits (POSITIF / NEGATIF / SPAM)
et logge toutes les expériences dans MLflow.

Usage:
    python src/train.py
    python src/train.py --n_features 5000 --C 0.5
"""

import argparse
import os
import json
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score,
    confusion_matrix
)

# ─── PARAMÈTRES ─────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Entraîne le classificateur de reviews")
    parser.add_argument("--n_features", type=int, default=3000,
                        help="Nombre de features TF-IDF")
    parser.add_argument("--max_df", type=float, default=0.9,
                        help="max_df pour TF-IDF (ignore les termes trop fréquents)")
    parser.add_argument("--min_df", type=int, default=1,
                        help="min_df pour TF-IDF")
    parser.add_argument("--C", type=float, default=1.0,
                        help="Paramètre de régularisation LogisticRegression")
    parser.add_argument("--max_iter", type=int, default=1000,
                        help="Nombre max d'itérations LogisticRegression")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Proportion du jeu de test")
    return parser.parse_args()


# ─── CHARGEMENT DES DONNÉES ──────────────────────────────────────────────────
def load_data(path="data/reviews.csv"):
    """Charge et valide le dataset."""
    df = pd.read_csv(path)
    assert "text" in df.columns and "label" in df.columns, \
        "Le CSV doit avoir les colonnes 'text' et 'label'"
    print(f"Dataset chargé : {len(df)} exemples")
    print(df["label"].value_counts())
    return df["text"], df["label"]


# ─── PIPELINE SKLEARN ────────────────────────────────────────────────────────
def build_pipeline(n_features, max_df, min_df, C, max_iter):
    """
    Construit le pipeline ML :
    TF-IDF → Logistic Regression

    TF-IDF : transforme le texte en vecteurs numériques
             en pondérant chaque mot par sa fréquence
             (rare dans un doc mais fréquent globalement = important)
    LR     : modèle linéaire rapide, interprétable, efficace sur du texte
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=3000,    # garde seulement les 3000 mots les plus importants
            max_df=0.9,           # ignore les mots présents dans +90% des textes (trop communs)
            min_df=1,             # garde un mot même s'il n'apparaît qu'une fois
            ngram_range=(1, 2),   # ← IMPORTANT : unigrammes ET bigrammes
            sublinear_tf=True,    # utilise log(tf) au lieu de tf brut → réduit l'effet des mots très fréquents
        )),
#"produit" apparaît dans presque tous les avis positifs, négatifs ET spam → TF-IDF l'ignorerait
#"promo", "cliquez" n'apparaissent que dans les SPAM → très discriminant, TF-IDF le garde

        ("clf", LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver="lbfgs",
            random_state=42,
        ))
    ])


# ─── ENTRAÎNEMENT + MLFLOW ───────────────────────────────────────────────────
def train(args):
    # Configurer l'expérience MLflow
    mlflow.set_experiment("ecommerce-review-classifier")

    X, y = load_data()

    # Split train/test (stratifié pour garder les proportions de classes)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    print(f"\nTrain : {len(X_train)} | Test : {len(X_test)}")

    # Démarrer un run MLflow — tout ce qui est loggé ici est tracé
    with mlflow.start_run():

        # 1. LOGGER LES PARAMÈTRES
        mlflow.log_params({
            "n_features": args.n_features,
            "max_df": args.max_df,
            "min_df": args.min_df,
            "C": args.C,
            "max_iter": args.max_iter,
            "test_size": args.test_size,
            "model_type": "TF-IDF + LogisticRegression",
        })

        # 2. ENTRAÎNER
        pipeline = build_pipeline(
            args.n_features, args.max_df, args.min_df, args.C, args.max_iter
        )
        pipeline.fit(X_train, y_train)

        # 3. ÉVALUER
        y_pred = pipeline.predict(X_test)
        y_train_pred = pipeline.predict(X_train)

        test_accuracy = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average="weighted")
        train_accuracy = accuracy_score(y_train, y_train_pred)

        # Cross-validation sur le train set
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="f1_weighted")
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        # 4. LOGGER LES MÉTRIQUES
        mlflow.log_metrics({
            "test_accuracy": test_accuracy,
            "test_f1_weighted": test_f1,
            "train_accuracy": train_accuracy,
            "cv_f1_mean": cv_mean,
            "cv_f1_std": cv_std,
        })

        # 5. AFFICHER LE RAPPORT
        print("\n" + "="*50)
        print("RÉSULTATS")
        print("="*50)
        print(f"Test Accuracy : {test_accuracy:.4f}")
        print(f"Test F1 (weighted) : {test_f1:.4f}")
        print(f"Train Accuracy : {train_accuracy:.4f}")
        print(f"CV F1 : {cv_mean:.4f} ± {cv_std:.4f}")
        print("\nClassification Report :")
        print(classification_report(y_test, y_pred))

        # 6. SAUVEGARDER LE MODÈLE AVEC MLFLOW
        mlflow.sklearn.log_model(pipeline, "review_classifier")
        print(f"\nModèle loggé dans MLflow (run ID: {mlflow.active_run().info.run_id})")

        # 7. SAUVEGARDER EN LOCAL (pour FastAPI)
        os.makedirs("models", exist_ok=True)
        with open("models/pipeline.pkl", "wb") as f:
            pickle.dump(pipeline, f)

        # Sauvegarder les classes pour l'API
        classes = list(pipeline.classes_)
        with open("models/classes.json", "w") as f:
            json.dump(classes, f)

        print(f"Modèle sauvegardé : models/pipeline.pkl")
        print(f"Classes : {classes}")

        # 8. LOGGER UN ARTEFACT (rapport texte)
        report = classification_report(y_test, y_pred)
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

        return pipeline, test_f1


if __name__ == "__main__":
    args = parse_args()
    model, f1 = train(args)
    print(f"\nEntraînement terminé. F1 score : {f1:.4f}")
    print("Lancer l'UI MLflow avec : mlflow ui")
    print("Puis ouvrir http://localhost:5000 pour voir les expériences")
