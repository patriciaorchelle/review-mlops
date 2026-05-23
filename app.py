import pickle, json, os
from datetime import datetime
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = "models/pipeline.pkl"
CLASSES_PATH = "models/classes.json"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modele introuvable : {MODEL_PATH}\nLance : python src/train.py")
    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)
    with open(CLASSES_PATH, "r") as f:
        classes = json.load(f)
    print(f"Modele charge depuis {MODEL_PATH}, classes : {classes}")
    return pipeline, classes

pipeline, CLASSES = load_model()

class ReviewInput(BaseModel):
    text: str
    model_config = {"json_schema_extra": {"example": {"text": "Tres bonne qualite, je recommande !"}}}

class BatchInput(BaseModel):
    texts: List[str]

class PredictionOutput(BaseModel):
    text: str
    label: str
    confidence: float
    probabilities: dict
    timestamp: str

class BatchOutput(BaseModel):
    predictions: List[PredictionOutput]
    count: int

app = FastAPI(
    title="Ecommerce Review Classifier API",
    description="Classifie les avis produits en POSITIF, NEGATIF ou SPAM",
    version="1.0.0",
)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "classes": CLASSES,
        "timestamp": datetime.utcnow().isoformat(),
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(review: ReviewInput):
    if not review.text.strip():
        raise HTTPException(status_code=400, detail="Le texte ne peut pas etre vide")
    label = pipeline.predict([review.text])[0]
    probas = pipeline.predict_proba([review.text])[0]
    confidence = float(max(probas))
    proba_dict = {cls: round(float(p), 4) for cls, p in zip(CLASSES, probas)}
    return PredictionOutput(
        text=review.text, label=label,
        confidence=round(confidence, 4),
        probabilities=proba_dict,
        timestamp=datetime.utcnow().isoformat(),
    )

@app.post("/predict/batch", response_model=BatchOutput)
def predict_batch(batch: BatchInput):
    if not batch.texts:
        raise HTTPException(status_code=400, detail="La liste de textes est vide")
    if len(batch.texts) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 textes par requete")
    labels = pipeline.predict(batch.texts)
    probas_all = pipeline.predict_proba(batch.texts)
    predictions = []
    for text, label, probas in zip(batch.texts, labels, probas_all):
        proba_dict = {cls: round(float(p), 4) for cls, p in zip(CLASSES, probas)}
        predictions.append(PredictionOutput(
            text=text, label=label,
            confidence=round(float(max(probas)), 4),
            probabilities=proba_dict,
            timestamp=datetime.utcnow().isoformat(),
        ))
    return BatchOutput(predictions=predictions, count=len(predictions))

@app.get("/model/info")
def model_info():
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    return {
        "model_type": "TF-IDF + Logistic Regression",
        "classes": CLASSES,
        "n_features": tfidf.max_features,
        "vocabulary_size": len(tfidf.vocabulary_) if hasattr(tfidf, "vocabulary_") else "not fitted",
        "regularization_C": clf.C,
    }
