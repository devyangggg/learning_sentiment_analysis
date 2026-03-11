# Sentiment Analysis API

A simple REST API that predicts movie review sentiment (positive/negative) using an LSTM model trained on the IMDB dataset.


### 1. Start the API server

```bash
uvicorn backend.app:app --reload
```

The server will start at `http://127.0.0.1:8000`.

## Usage

### Predict sentiment

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was amazing"}'
```

**Response:**

```json
{
  "sentiment": "positive",
  "confidence": 0.93
}
```

### Health check

```bash
curl http://127.0.0.1:8000/health
```

### API docs

FastAPI auto-generates interactive docs at `http://127.0.0.1:8000/docs`.
