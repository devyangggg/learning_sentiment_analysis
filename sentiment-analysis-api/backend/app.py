
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from .model import load_model
from .preprocess import load_vocab, preprocess_text


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "sentiment_model.pt")
VOCAB_PATH = os.path.join(BASE_DIR, "..", "models", "vocab.json")

model = None
vocab = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, vocab

    vocab = load_vocab(VOCAB_PATH)
    model = load_model(MODEL_PATH, vocab_size=len(vocab))

    print(f"Model loaded  — vocab size: {len(vocab)}")
    yield  # server runs here
    # cleanup (nothing to do)


app = FastAPI(
    title="Sentiment Analysis API",
    description="Predict sentiment (positive / negative) for text.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    sentiment: str
    confidence: float

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):

    tokens, mask = preprocess_text(request.text, vocab)

    with torch.no_grad():
        logits = model(tokens, mask).squeeze(1)
        probability = torch.sigmoid(logits).item()


    sentiment = "positive" if probability >= 0.5 else "negative"
    confidence = probability if sentiment == "positive" else 1 - probability

    return PredictResponse(sentiment=sentiment, confidence=round(confidence, 4))
