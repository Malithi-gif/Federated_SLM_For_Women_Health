from __future__ import annotations

import gc
import json
import os
import threading
from pathlib import Path
from typing import Any

# Set CPU limits before importing PyTorch.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from peft import PeftModel
from pydantic import BaseModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    # This may already have been configured by the runtime.
    pass


# ---------------------------------------------------------------------
# Paths and model configuration
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
ADAPTER_DIR = ROOT / "models" / "dp_adapter"
PREPROCESSOR_PATH = ROOT / "preprocessor.json"

BASE_MODEL = "distilbert/distilbert-base-uncased"
DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------
# Validate required files
# ---------------------------------------------------------------------

if not PREPROCESSOR_PATH.exists():
    raise FileNotFoundError(
        f"Preprocessor file was not found: {PREPROCESSOR_PATH}"
    )

if not ADAPTER_DIR.exists():
    raise FileNotFoundError(
        f"Adapter directory was not found: {ADAPTER_DIR}"
    )

if not (ADAPTER_DIR / "adapter_config.json").exists():
    raise FileNotFoundError(
        f"adapter_config.json was not found in {ADAPTER_DIR}"
    )

if not (ADAPTER_DIR / "adapter_model.safetensors").exists():
    raise FileNotFoundError(
        f"adapter_model.safetensors was not found in {ADAPTER_DIR}"
    )


# ---------------------------------------------------------------------
# Load preprocessing metadata
# ---------------------------------------------------------------------

with PREPROCESSOR_PATH.open("r", encoding="utf-8") as file:
    PREPROCESSOR = json.load(file)

ID2LABEL = {
    int(key): value
    for key, value in PREPROCESSOR["id2label"].items()
}

LABEL2ID = {
    label: label_id
    for label_id, label in ID2LABEL.items()
}

NUMERIC_MEDIANS = PREPROCESSOR["numeric_medians"]
CATEGORICAL_DEFAULTS = PREPROCESSOR["categorical_fill_values"]
NUMERIC_FEATURES = PREPROCESSOR["numeric_features"]
CATEGORICAL_FEATURES = PREPROCESSOR["categorical_features"]

MAX_LENGTH = int(
    PREPROCESSOR.get("max_seq_length", 384)
)


# ---------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------

def clean_payload(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}

    for feature_name in NUMERIC_FEATURES:
        value = payload.get(feature_name)

        if value is None:
            value = NUMERIC_MEDIANS[feature_name]

        cleaned[feature_name] = float(value)

    for feature_name in CATEGORICAL_FEATURES:
        value = payload.get(feature_name)

        if value is None or str(value).strip() == "":
            value = CATEGORICAL_DEFAULTS[feature_name]

        cleaned[feature_name] = str(value)

    return cleaned


def build_text(row: dict[str, Any]) -> str:
    """
    This feature-to-text template should match the template used during
    DistilBERT training. Replace it with the exact training function when
    that function is available.
    """

    return (
        f"study interval: {row['study_interval']}; "
        f"is weekend: {row['is_weekend']}; "
        f"day in study: {row['day_in_study']}; "
        f"lh: {row['lh']}; "
        f"estrogen: {row['estrogen']}; "
        f"pdg: {row['pdg']}; "
        f"flow volume: {row['flow_volume']}; "
        f"flow color: {row['flow_color']}; "
        f"headaches: {row['headaches']}; "
        f"cramps: {row['cramps']}; "
        f"sore breasts: {row['sorebreasts']}; "
        f"fatigue: {row['fatigue']}; "
        f"sleep issue: {row['sleepissue']}; "
        f"mood swing: {row['moodswing']}; "
        f"food cravings: {row['foodcravings']}; "
        f"indigestion: {row['indigestion']}; "
        f"bloating: {row['bloating']}."
    )


# ---------------------------------------------------------------------
# Lazy model loading
# ---------------------------------------------------------------------
#
# The model is not loaded while FastAPI starts. This allows Render to
# open its required port before model initialization.
#
# Important: the first prediction will take longer because the model
# is loaded at that time.
# ---------------------------------------------------------------------

tokenizer: AutoTokenizer | None = None
model: PeftModel | None = None
model_lock = threading.Lock()


def load_model() -> tuple[AutoTokenizer, PeftModel]:
    global tokenizer
    global model

    if tokenizer is not None and model is not None:
        return tokenizer, model

    with model_lock:
        if tokenizer is not None and model is not None:
            return tokenizer, model

        loaded_tokenizer = AutoTokenizer.from_pretrained(
            ADAPTER_DIR,
            local_files_only=True,
            use_fast=True,
        )

        base_model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                BASE_MODEL,
                num_labels=len(ID2LABEL),
                id2label=ID2LABEL,
                label2id=LABEL2ID,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32,
            )
        )

        loaded_model = PeftModel.from_pretrained(
            base_model,
            ADAPTER_DIR,
            is_trainable=False,
            low_cpu_mem_usage=True,
        )

        loaded_model.to(DEVICE)
        loaded_model.eval()

        tokenizer = loaded_tokenizer
        model = loaded_model

        gc.collect()

        return tokenizer, model


# ---------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------

app = FastAPI(
    title="DistilBERT Menstrual Phase API",
    description=(
        "Federated DistilBERT differential-privacy adapter for "
        "menstrual phase classification."
    ),
    version="1.1.0",
)


allowed_origins = [
    "https://malithi-gif.github.io",
    "https://malithi-gif.github.io/"
    "Federated_SLM_For_Women_Health",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]

extra_origin = os.getenv("FRONTEND_ORIGIN")

if extra_origin:
    allowed_origins.append(extra_origin.rstrip("/"))

allowed_origins = list(dict.fromkeys(allowed_origins))


app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": "DistilBERT federated DP adapter",
        "model_loaded": model is not None,
        "documentation": "/docs",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(inputs: PredictionInput) -> dict[str, Any]:
    try:
        active_tokenizer, active_model = load_model()

        payload = inputs.model_dump()
        cleaned = clean_payload(payload)
        text = build_text(cleaned)

        encoded = active_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=False,
            max_length=MAX_LENGTH,
        )

        encoded = {
            name: tensor.to(DEVICE)
            for name, tensor in encoded.items()
        }

        with torch.inference_mode():
            outputs = active_model(**encoded)
            logits = outputs.logits[0]
            probabilities_tensor = torch.softmax(
                logits,
                dim=-1,
            )

        probabilities = (
            probabilities_tensor
            .detach()
            .cpu()
            .tolist()
        )

        best_id = int(
            torch.argmax(
                probabilities_tensor
            ).item()
        )

        probability_output = {
            ID2LABEL[index]: float(probability)
            for index, probability in enumerate(probabilities)
        }

        del encoded
        del outputs
        del logits
        del probabilities_tensor

        gc.collect()

        return {
            "predicted_phase": ID2LABEL[best_id],
            "confidence": float(probabilities[best_id]),
            "probabilities": probability_output,
            "model": "DistilBERT federated DP adapter",
            "privacy_setting": "client-level differential privacy",
            "input_text": text,
            "disclaimer": (
                "Research demonstration only. "
                "This output is not medical advice."
            ),
        }

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Required model file is missing: {error}",
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}",
        ) from error