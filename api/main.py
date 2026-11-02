"""
API de classification des demandes de services (texte libre -> service / sous-service).

Lance les deux modeles FastText entraines par train_model.py et expose un
endpoint de prediction. Documentation interactive auto-generee par FastAPI
sur /docs (utilisable directement depuis un navigateur, sans ecrire de code).

Usage:
    uvicorn api.main:app --reload --port 8000
Puis ouvrir http://127.0.0.1:8000/docs
"""
import csv
import os
from collections import defaultdict
from typing import List

import numpy as np

# fasttext 0.9.2 appelle en interne np.array(probs, copy=False), une syntaxe
# devenue stricte avec numpy>=2.0 (leve ValueError au lieu de copier si besoin).
# On rend ce raccourci tolerant plutot que de figer numpy<2 (requis par ailleurs
# par pandas), pour ne pas avoir deux environnements incompatibles.
_original_np_array = np.array


def _tolerant_array(*args, **kwargs):
    if kwargs.get("copy") is False:
        kwargs.pop("copy")
        return np.asarray(*args, **kwargs)
    return _original_np_array(*args, **kwargs)


np.array = _tolerant_array

import fasttext  # noqa: E402  (doit etre importe apres le monkeypatch ci-dessus)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TAXONOMY_PATH = os.path.join(BASE_DIR, "data", "taxonomy.csv")

app = FastAPI(
    title="Classification des demandes de services",
    description=(
        "Classifie le texte libre d'une demande de travaux/prestation en "
        "categorie de service (250 classes) et de sous-service (750 classes), "
        "via deux modeles FastText entraines sur des donnees synthetiques."
    ),
    version="1.0.0",
)

_models = {}
_service_to_sous = defaultdict(set)


def _label_to_text(label: str) -> str:
    return label.replace("__label__", "", 1).replace("_", " ")


@app.on_event("startup")
def load_models():
    for name in ("service", "sous_service"):
        path = os.path.join(MODELS_DIR, f"{name}.bin")
        if not os.path.exists(path):
            raise RuntimeError(
                f"Modele introuvable: {path}. Lancez d'abord generate_taxonomy.py, "
                "generate_synthetic_data.py, data_augmentation.py puis train_model.py."
            )
        _models[name] = fasttext.load_model(path)

    if os.path.exists(TAXONOMY_PATH):
        with open(TAXONOMY_PATH, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                _service_to_sous[row["service"]].add(row["sous_service"])


class PredictRequest(BaseModel):
    description: str = Field(
        ...,
        min_length=3,
        description="Texte libre decrivant la demande (ex: 'je cherche un plombier pour reparer ma chaudiere en urgence').",
        examples=["Bonjour, je cherche quelqu'un pour reparer ma chaudiere le plus vite possible."],
    )
    top_k: int = Field(3, ge=1, le=10, description="Nombre de suggestions par niveau (service / sous-service).")


class Prediction(BaseModel):
    label: str
    confidence: float


class PredictResponse(BaseModel):
    description: str
    service: List[Prediction]
    sous_service: List[Prediction]


def _predict(name: str, text: str, k: int) -> List[Prediction]:
    model = _models[name]
    labels, probs = model.predict(text.replace("\n", " "), k=k)
    return [
        Prediction(label=_label_to_text(label), confidence=round(float(p), 4))
        for label, p in zip(labels, probs)
    ]


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(_models.keys())}


@app.get("/services")
def list_services():
    return {"count": len(_service_to_sous), "services": sorted(_service_to_sous.keys())}


@app.get("/services/{service}/sous_services")
def list_sous_services(service: str):
    if service not in _service_to_sous:
        raise HTTPException(status_code=404, detail=f"service inconnu: {service}")
    return {"service": service, "sous_services": sorted(_service_to_sous[service])}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    return PredictResponse(
        description=req.description,
        service=_predict("service", req.description, req.top_k),
        sous_service=_predict("sous_service", req.description, req.top_k),
    )
