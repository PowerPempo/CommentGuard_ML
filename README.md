# 🛡️ CommentGuard ML

> AI-powered system for automatic detection of toxic, spammy, and inappropriate comments.

Built as an end-to-end ML project — from data preprocessing and model training to REST API deployment with Docker.

---

## 📌 Overview

Online platforms need scalable, reliable moderation. CommentGuard explores how classical NLP techniques can handle multilabel text classification in real-world scenarios, with a focus on the full ML lifecycle — not just model accuracy.

**Primary goal:** demonstrate a structured ML development process, from problem formulation to deployment.

---

## 🧠 ML Approach

**Problem type:** Multilabel text classification — a single comment can belong to multiple categories simultaneously.

**Labels:**

| Label | Description |
|---|---|
| `toxic` | General toxic language |
| `severe_toxic` | Extreme toxicity |
| `obscene` | Obscene content |
| `threat` | Threatening language |
| `insult` | Insulting language |
| `identity_hate` | Hate speech targeting identity |

**Baseline model:**
- TF-IDF (unigrams + bigrams, 50k features)
- Logistic Regression with One-vs-Rest strategy
- Per-label threshold tuning on a validation set

---

## 📊 Model Performance

Evaluated on a held-out test set (20% of data) with per-label thresholds tuned on a validation set.

### Per-label thresholds & F1 scores (validation set)

| Label | Threshold | F1 |
|---|---|---|
| toxic | 0.80 | 0.7643 |
| severe_toxic | 0.65 | 0.5082 |
| obscene | 0.70 | 0.8118 |
| threat | 0.85 | 0.4844 |
| insult | 0.65 | 0.6991 |
| identity_hate | 0.85 | 0.4643 |

### Classification report (test set)

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| toxic | 0.84 | 0.70 | 0.76 | 3029 |
| severe_toxic | 0.33 | 0.68 | 0.44 | 307 |
| obscene | 0.81 | 0.81 | 0.81 | 1678 |
| threat | 0.47 | 0.44 | 0.46 | 99 |
| insult | 0.66 | 0.76 | 0.71 | 1551 |
| identity_hate | 0.45 | 0.47 | 0.46 | 290 |
| **micro avg** | 0.72 | 0.73 | **0.72** | 6954 |
| **macro avg** | 0.59 | 0.64 | **0.61** | 6954 |
| **weighted avg** | 0.75 | 0.73 | **0.73** | 6954 |

> ⚠️ Lower F1 on rare labels (`threat`, `identity_hate`) is expected for a TF-IDF baseline — DistilBERT/RoBERTa fine-tuning is planned to address this.

---

## ⚙️ Tech Stack

- **Python 3.11**
- **Pandas, NumPy, Scikit-learn**
- **FastAPI** — REST API
- **Docker** — containerized deployment

---

## 🚀 Quick Start

### Run with Docker

```bash
# Build the image
docker build -t commentguard .

# Run the container
docker run -p 8000:8000 commentguard
```

Open the interactive API docs at: `http://localhost:8000/docs`

### Run locally

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Train the model
python src/models/train.py

# Start the API
fastapi dev src/models/api/main.py
```

---

## 📡 API

### `POST /predict`

**Request:**
```json
{
  "text": "you are stupid"
}
```

**Response:**
```json
{
  "is_banned": true,
  "confidence": 0.82,
  "labels": {
    "toxic": { "confidence": 0.82, "flagged": true },
    "severe_toxic": { "confidence": 0.12, "flagged": false },
    "obscene": { "confidence": 0.09, "flagged": false },
    "threat": { "confidence": 0.03, "flagged": false },
    "insult": { "confidence": 0.74, "flagged": true },
    "identity_hate": { "confidence": 0.02, "flagged": false }
  }
}
```

### `GET /health`

```json
{ "status": "ok" }
```

---

## 📁 Project Structure

```
CommentGuard_ML/
├── src/
│   ├── models/
│   │   ├── api/
│   │   │   ├── main.py          # FastAPI app
│   │   │   ├── model_loader.py  # Model loading & inference
│   │   │   └── schemas.py       # Pydantic schemas
│   │   └── train.py             # Training pipeline
│   └── data/
│       └── processed/           # Preprocessed datasets
├── models/
│   ├── CommentGuard_ML.pkl      # Trained model (not tracked in git)
│   ├── labels.json              # Active label names
│   └── thresholds.json          # Per-label thresholds
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🗺️ Roadmap

- [x] Data preprocessing
- [x] Multilabel baseline model (TF-IDF + LogReg + OvR)
- [x] Per-label threshold tuning
- [x] FastAPI inference service
- [x] Docker deployment
- [ ] DistilBERT / RoBERTa fine-tuning
- [ ] MLflow experiment tracking
- [ ] Unit tests
- [ ] Model monitoring & retraining

---

## 📖 ML Design Decisions

**Why One-vs-Rest?** Each label is independent — a comment can be both toxic and obscene at the same time. OvR trains a separate binary classifier per label, which is the correct framing for multilabel problems.

**Why per-label thresholds?** The default threshold of 0.5 is rarely optimal, especially for imbalanced classes. Rare labels like `threat` need lower thresholds to catch positive cases — tuning per label on a validation set gives a meaningful F1 improvement.

**Why TF-IDF as baseline?** Fast to train, interpretable, and provides a strong reference point. Any future deep learning model needs to beat this baseline to justify the added complexity.
