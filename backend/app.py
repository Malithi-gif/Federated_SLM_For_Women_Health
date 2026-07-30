from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
ADAPTER_DIR = ROOT / "models" / "dp_adapter"
PREPROCESSOR_PATH = ROOT / "preprocessor.json"
BASE_MODEL = "distilbert/distilbert-base-uncased"

with PREPROCESSOR_PATH.open("r", encoding="utf-8") as f:
    PREPROCESSOR = json.load(f)

ID2LABEL = {int(k): v for k, v in PREPROCESSOR["id2label"].items()}
NUMERIC_MEDIANS = PREPROCESSOR["numeric_medians"]
CATEGORICAL_DEFAULTS = PREPROCESSOR["categorical_fill_values"]
MAX_LENGTH = int(PREPROCESSOR["max_seq_length"])

class PredictionInput(BaseModel):
    study_interval: float | None = None
    is_weekend: bool | str | None = None
    day_in_study: float | None = None
    lh: float | None = None
    estrogen: float | None = None
    pdg: float | None = None
    flow_volume: str | None = None
    flow_color: str | None = None
    headaches: str | None = None
    cramps: str | None = None
    sorebreasts: str | None = None
    fatigue: str | None = None
    sleepissue: str | None = None
    moodswing: str | None = None
    foodcravings: str | None = None
    indigestion: str | None = None
    bloating: str | None = None

def clean_payload(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for name in PREPROCESSOR["numeric_features"]:
        value = payload.get(name)
        cleaned[name] = NUMERIC_MEDIANS[name] if value is None else value

    for name in PREPROCESSOR["categorical_features"]:
        value = payload.get(name)
        if value is None or str(value).strip() == "":
            value = CATEGORICAL_DEFAULTS[name]
        cleaned[name] = str(value)
    return cleaned

def build_text(row: dict[str, Any]) -> str:
    """
    IMPORTANT:
    This template must match the feature-to-text function used during training.
    Replace this function with the exact build_text/row_to_text function from
    the training script when available.
    """
    return (
        f"study interval: {row['study_interval']}; "
        f"is weekend: {row['is_weekend']}; "
        f"day in study: {row['day_in_study']}; "
        f"lh: {row['lh']}; estrogen: {row['estrogen']}; pdg: {row['pdg']}; "
        f"flow volume: {row['flow_volume']}; flow color: {row['flow_color']}; "
        f"headaches: {row['headaches']}; cramps: {row['cramps']}; "
        f"sore breasts: {row['sorebreasts']}; fatigue: {row['fatigue']}; "
        f"sleep issue: {row['sleepissue']}; mood swing: {row['moodswing']}; "
        f"food cravings: {row['foodcravings']}; indigestion: {row['indigestion']}; "
        f"bloating: {row['bloating']}."
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=4,
    id2label=ID2LABEL,
    label2id={v: k for k, v in ID2LABEL.items()},
)
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.to(device)
model.eval()

app = FastAPI(title="DistilBERT Menstrual Phase API", version="1.0.0")

allowed_origins = [
    "https://malithi-gif.github.io",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]
extra_origin = os.getenv("FRONTEND_ORIGIN")
if extra_origin:
    allowed_origins.append(extra_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "model": "DistilBERT DP adapter"}

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy", "device": str(device)}

@app.post("/predict")
def predict(inputs: PredictionInput) -> dict[str, Any]:
    try:
        cleaned = clean_payload(inputs.model_dump())
        text = build_text(cleaned)
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.inference_mode():
            logits = model(**encoded).logits[0]
            probabilities = torch.softmax(logits, dim=-1).cpu().tolist()

        best_id = int(max(range(len(probabilities)), key=probabilities.__getitem__))
        return {
            "predicted_phase": ID2LABEL[best_id],
            "confidence": probabilities[best_id],
            "probabilities": {
                ID2LABEL[i]: probabilities[i] for i in range(len(probabilities))
            },
            "model": "DistilBERT federated DP adapter",
            "input_text": text,
            "disclaimer": "Research demonstration only. Not medical advice.",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
