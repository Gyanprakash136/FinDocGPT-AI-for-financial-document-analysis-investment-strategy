# FinDocGPT — From Financial Docs to Decisions (Stage 1–3)

**AkashX.ai Hackathon project**

Turn unstructured financial documents + market data into **actionable** buy/sell decisions.

- **Stage 1 – Insights & Analysis:** Upload PDFs → RAG Q&A → Sentiment → Anomaly detection  
- **Stage 2 – Forecasting:** Yahoo Finance history → Prophet + LSTM + XGBoost → Ensemble forecast  
- **Stage 3 – Strategy:** BUY/SELL/HOLD signals with confidence, holding period, and an interactive Plotly dashboard

---

## Table of Contents
- [Demo Screenshots](#demo-screenshots)
- [What We Built](#what-we-built)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [How It Works (Stages 1–3)](#how-it-works-stages-1–3)
- [Configuration Knobs](#configuration-knobs)
- [API Endpoints](#api-endpoints)
- [Troubleshooting](#troubleshooting)
- [Tech Stack](#tech-stack)
- [Requirements.txt](#requirementstxt)
- [Roadmap](#roadmap)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Demo Screenshots

> Place images in `screenshots/` as `1.png` … `12.png`. They’ll render automatically below.

| #  | Feature                                       | Image                           |
|----|-----------------------------------------------|---------------------------------|
| 1  | Home / Health check                           | ![1](screenshots/1.png)         |
| 2  | PDF Upload (Stage 1)                          | ![2](screenshots/2.png)         |
| 3  | Processing Status                             | ![3](screenshots/3.png)         |
| 4  | Q&A over PDF chunks                           | ![4](screenshots/4.png)         |
| 5  | Sentiment (session or custom text)            | ![5](screenshots/5.png)         |
| 6  | Anomaly Detection (YoY/Margins)               | ![6](screenshots/6.png)         |
| 7  | Forecast Inputs (Stage 2)                     | ![7](screenshots/7.png)         |
| 8  | Strategy Chart: Price + Ensemble              | ![8](screenshots/8.png)         |
| 9  | Signal Timeline (BUY/HOLD/SELL)               | ![9](screenshots/9.png)         |
| 10 | Expected Return & Confidence                  | ![10](screenshots/10.png)       |
| 11 | Model Metrics Table                           | ![11](screenshots/11.png)       |
| 12 | Downloadable CSVs + Embedded Chart            | ![12](screenshots/12.png)       |

---

## What We Built

**Problem:** Financial reports and market data are fragmented. Analysts lose time stitching insights from PDFs, price history, and sentiment—then translating that into trade decisions.

**Solution:** FinDocGPT ingests PDFs, answers questions grounded in the upload, scores sentiment, flags anomalies, predicts prices with an ensemble of models, and outputs **clear BUY/SELL/HOLD signals** with confidence and holding period.

**Who benefits:** Equity analysts, PMs, fintech builders, and students learning applied AI in finance.

**What’s impressive:**  
- End-to-end flow from **document → insight → forecast → trade** in one UI  
- Fast RAG with persistent embeddings (ChromaDB)  
- Ensemble forecasting + transparent, interactive Plotly dashboard  
- Practical strategy layer with CSV exports and embeddable chart

---

## Architecture

**Frontend:** Streamlit (`frontend.py`)  
**Backend:** FastAPI (`main.py`)
- **RAG store:** ChromaDB (persistent local DB)
- **Embeddings:** Sentence-Transformers `all-MiniLM-L6-v2`
- **LLM helper:** Gemini 2.5 Flash (structured JSON prompts for sentiment/anomaly)
- **Forecasting:** Prophet + LSTM (Keras/TensorFlow) + XGBoost (feature-engineered lags/stats)
- **Strategy:** Threshold rules → BUY/SELL/HOLD + confidence + holding period
- **Static:** Interactive Plotly HTML + CSV files, served via `/static`

---

## Project Structure

/ (repo root)
├─ FDocgpt_backend.py # FastAPI backend (Stage 1–3)
├─ FDocgpt_streamLit.py # Streamlit UI (Stage 1–3)
├─ outputs/ # Plotly HTML & CSVs (served at /static)
├─ db/ # Chroma persistent store
├─ temp_files/ # Uploaded PDFs (transient)
├─ screenshots/ # 1.png ... 12.png (UI walkthrough)
├─ .env # GEMINI_API_KEY=...
└─ requirements.txt


---

## Quickstart

### 1) Create environment (Python 3.10 recommended)
```bash
# venv
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# OR conda
conda create -n findocgpt python=3.10 -y
conda activate findocgpt
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt


GEMINI_API_KEY=YOUR_GOOGLE_AI_STUDIO_KEY


# Windows (PowerShell)
$env:KERAS_BACKEND="tensorflow"
# macOS/Linux
export KERAS_BACKEND=tensorflow


3) Configure environment
Create .env:

GEMINI_API_KEY=YOUR_GOOGLE_AI_STUDIO_KEY
Set Keras backend (recommended, esp. on Windows):

# Windows (PowerShell)
$env:KERAS_BACKEND="tensorflow"
# macOS/Linux
export KERAS_BACKEND=tensorflow
4) Run the backend

uvicorn main:app --reload --port 8000
Health check: http://localhost:8000/

5) Run the frontend

streamlit run frontend.py
Use the Streamlit URL printed in your terminal.

How It Works (Stages 1–3)
Stage 1 — Insights & Analysis
Upload PDF → chunk with PyMuPDF → embed via all-MiniLM-L6-v2 → store in ChromaDB (session-scoped).

Q&A → semantic retrieval and a concise, context-only answer.

Sentiment → Gemini returns strict JSON {label, polarity, confidence, rationale}; falls back to rule-based lexicon if API fails.

Anomaly detection → Gemini extracts metrics (YoY/QoQ, margins, EPS) as JSON; regex fallback for common patterns.

Stage 2 — Forecasting
Data fetch → Yahoo Finance, auto-adjusted prices.

Feature engineering → TA indicators (RSI, MACD, BB), lags/rolling stats, cyclical time features.

Models

Prophet: seasonality + trend

LSTM: sequence modeling (close or multi-features)

XGBoost: lags & rolling features

Ensemble → weighted mean of valid forecasts → horizon N.

Dashboard → historical vs. ensemble path, model metrics table.

Stage 3 — Strategy
Signals: Compute stepwise expected returns; thresholds default to BUY > +2%, SELL < −2%.

Confidence: Scales with move magnitude, capped at 100%.

Holding period: Simple forward scan—stop on signal flip or low-move; default horizon if none found.

Outputs: First recommended action, CSVs, and an interactive Plotly chart (embedded in Streamlit, downloadable from /static).

Configuration Knobs
Stage 1

CHUNK_SIZE (default 300)

top_k retrieval (frontend control)

Stage 2

horizon (5–120 days; frontend control)

LSTM: LSTM_WINDOW, LSTM_EPOCHS, TEST_SPLIT_RATIO (see Config in main.py)

Stage 3

Buy/Sell thresholds: Config.BUY_THRESHOLD = 0.02, SELL_THRESHOLD = -0.02

Adjust in main.py for more/less aggressive signals

API Endpoints
Health
pgsql
Copy
Edit
GET / → {status, version, endpoints}
Stage 1 – Documents & Insights
bash
Copy
Edit
POST /upload-pdf/            # returns {session_id, task_id}
GET  /status/{task_id}       # processing state
POST /query/                 # {question, top_k, session_id}
POST /sentiment/             # {text? | session_id, top_k_chunks}
POST /anomaly/               # {text? | session_id, top_k_chunks, yoy_threshold, margin_bps_threshold}
Example:

curl -X POST http://localhost:8000/query/ \
  -H "Content-Type: application/json" \
  -d '{"question":"What was Q2 revenue?","top_k":8,"session_id":"YOUR_SESSION_ID"}'
Stage 2 + 3 – Forecast & Strategy

POST /forecast/run           # {ticker, start, end, horizon, outdir}
# returns:
# - chart_url (Plotly HTML under /static)
# - signals_csv_url, forecasts_csv_url
# - next_action {action,date,price,holding_days,confidence,reason}
# - metrics {...}
Example:

curl -X POST http://localhost:8000/forecast/run \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AAPL","start":"2024-06-01","end":"2025-06-01","horizon":30,"outdir":"outputs"}'

GET    /debug/sessions                 # list sessions + chunk counts
DELETE /debug/sessions/{session_id}    # delete a session’s vectors
Troubleshooting
1) Streamlit duplicate widget IDs
Give a unique key= to widgets that have the same label/params.

2) Keras/TensorFlow mismatches
Use the pinned versions below and set KERAS_BACKEND=tensorflow. Restart your shell after changes.

3) HuggingFace hub / transformers conflicts
Pinned trio is known-good:


sentence-transformers==2.2.2
transformers==4.30.2
huggingface_hub==0.14.1
4) Prophet on Windows
If install fails:

pip install --upgrade pip wheel setuptools
pip install cmdstanpy
5) oneDNN / TF warnings
To normalize numerics (optional):


TF_ENABLE_ONEDNN_OPTS=0
6) yfinance returns empty
Check date range, market holidays, and symbol correctness (uppercase). Try a longer range.

7) Ports already in use
Change ports:

uvicorn main:app --reload --port 8001
streamlit run frontend.py --server.port 8502
8) CORS
Front/back run on same machine by default. If needed, add CORS middleware to FastAPI.

Tech Stack
Frontend: Streamlit

Backend: FastAPI, Uvicorn

RAG: ChromaDB, Sentence-Transformers

LLM Helper: Gemini 2.5 Flash (JSON outputs)

Forecasting: Prophet, Keras/TensorFlow LSTM, XGBoost

Indicators: ta (RSI, MACD, BB, MFI, etc.)

Charts: Plotly

Requirements.txt
Copy into requirements.txt:

# Web + API
fastapi==0.110.0
uvicorn[standard]==0.30.1
python-multipart==0.0.9
pydantic==2.7.1
requests==2.32.3
python-dotenv==1.0.1

# RAG
chromadb==0.5.0
sentence-transformers==2.2.2
transformers==4.30.2
huggingface_hub==0.14.1
PyMuPDF==1.24.1

# ML / Forecasting
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.4.2
xgboost==2.0.3
ta==0.11.0
yfinance==0.2.40
prophet==1.1.5

# Deep Learning (pinned to avoid TF/Keras breakages)
tensorflow==2.15.0
keras==2.15.0
tf-keras==2.15.0

# Charts
plotly==5.20.0

# Frontend
streamlit==1.37.1
Roadmap
Broker/Exchange connector for paper/live trading

Risk overlays (vol targeting, drawdown guardrails)

Explainable signals (shapley on XGBoost features)

Multi-doc session merges & fine-grained citations

Evaluation harness with backtests and ablations

License
MIT — see LICENSE (or update as needed).

Acknowledgments
AkashX.ai for the challenge & dataset direction

Open-source ecosystem: FastAPI, Streamlit, ChromaDB, Sentence-Transformers, Prophet, TensorFlow/Keras, XGBoost, Plotly

Yahoo Finance for historical market data



If you want, I can generate a matching **`CONTRIBUTING.md`** and a minimal **architecture diagram
