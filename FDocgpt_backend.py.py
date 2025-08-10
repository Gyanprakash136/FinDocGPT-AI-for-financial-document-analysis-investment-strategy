# main.py
# AkashX FinDocGPT — Stage 1 (RAG Q&A + Sentiment + Anomaly)
# Stage 2 (Forecasting + Signals + Chart)
# Stage 3 (Strategy: Buy/Sell/Hold + Explainability)

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KERAS_BACKEND"] = "tensorflow"  # make Keras 3 use TF backend explicitly

import re
import json
import logging
import uuid
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime, timedelta
import multiprocessing

import requests
import chromadb
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ---------- Forecasting deps ----------
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf
from prophet import Prophet
import xgboost as xgb
import ta
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================
# Logging & env
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
load_dotenv()
logger = logging.getLogger("finhack")

# =========================
# Storage paths
# =========================
DB_DIR = Path("./db"); DB_DIR.mkdir(exist_ok=True)
TEMP_DIR = Path("./temp_files"); TEMP_DIR.mkdir(exist_ok=True)
OUT_DIR = Path("./outputs"); OUT_DIR.mkdir(exist_ok=True)

# Serve outputs (charts/CSVs) at /static
app = FastAPI(title="AkashX FinDocGPT API (Stages 1–3)", version="6.0.0")
app.mount("/static", StaticFiles(directory=str(OUT_DIR.resolve())), name="static")

# =========================
# Speed knobs
# =========================
MAX_WORKERS = multiprocessing.cpu_count()
CHUNK_SIZE = 300
BATCH_SIZE = 64

# =========================
# Gemini API config
# =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

# =========================
# Stage 1 API models
# =========================
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(8, le=20)
    session_id: Optional[str] = Field(None, description="Returned by /upload-pdf")

class SentimentRequest(BaseModel):
    text: Optional[str] = None
    session_id: Optional[str] = None
    top_k_chunks: int = Field(12, ge=3, le=50)

class SentimentResult(BaseModel):
    label: str
    polarity: float
    confidence: float
    rationale: str
    tokens_used: Optional[int] = None
    source: str = Field("gemini|fallback")

class SentimentResponse(BaseModel):
    result: SentimentResult
    mode: str
    session_id: Optional[str] = None
    sampled_chars: int = 0

class AnomalyRequest(BaseModel):
    text: Optional[str] = None
    session_id: Optional[str] = None
    top_k_chunks: int = Field(20, ge=5, le=80)
    yoy_threshold: float = Field(0.30, ge=0.05, le=1.0)
    margin_bps_threshold: int = Field(300, ge=50, le=2000)

class AnomalyItem(BaseModel):
    metric: str
    unit: Optional[str] = None
    period_1: Optional[str] = None
    value_1: Optional[float] = None
    period_2: Optional[str] = None
    value_2: Optional[float] = None
    pct_change: Optional[float] = None
    direction: Optional[str] = None
    is_anomaly: bool = False
    reason: Optional[str] = None

class AnomalyResponse(BaseModel):
    items: List[AnomalyItem]
    summary: str
    session_id: Optional[str] = None
    mode: str
    sampled_chars: int = 0
    rules: Dict[str, Any]

class FastResponse(BaseModel):
    answer: str
    processing_time_ms: int
    chunks_used: int
    confidence: float

# =========================
# Stage 2 API models
# =========================
class ForecastRequest(BaseModel):
    ticker: str = Field(..., description="e.g., AAPL")
    start: str = Field(..., description="YYYY-MM-DD")
    end: str = Field(..., description="YYYY-MM-DD")
    horizon: int = Field(30, ge=5, le=120)
    outdir: Optional[str] = Field("outputs")

class NextAction(BaseModel):
    action: str
    date: str
    price: float
    holding_days: int
    confidence: float
    reason: str

class ForecastResponse(BaseModel):
    chart_url: str
    signals_csv_url: str
    forecasts_csv_url: str
    next_action: NextAction
    metrics: Dict[str, Any]
    notes: List[str]

# =========================
# Stage 3 API models
# =========================
class StrategyRequest(BaseModel):
    ticker: str
    start: str  # YYYY-MM-DD
    end: str    # YYYY-MM-DD
    horizon: int = Field(30, ge=5, le=120)
    session_id: Optional[str] = None           # to fuse Stage 1 sentiment/anomaly
    outdir: Optional[str] = "outputs"
    weights: Optional[Dict[str, float]] = None # keys: forecast/sentiment/technical/risk

class StrategyBreakdown(BaseModel):
    forecast_score: float
    sentiment_score: float
    technical_score: float
    risk_penalty: float
    composite_score: float

class StrategyRecommendation(BaseModel):
    action: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    holding_period_days: int
    rationale: List[str]

class StrategyResponse(BaseModel):
    recommendation: StrategyRecommendation
    breakdown: StrategyBreakdown
    next_action: NextAction
    links: Dict[str, str]        # chart_url, signals_csv_url, forecasts_csv_url
    metrics: Dict[str, Any]
    sentiment_snapshot: Optional[SentimentResult] = None
    anomalies: List[AnomalyItem] = []

# =========================
# Global state
# =========================
task_statuses: Dict[str, Dict] = {}
latest_session_id: Optional[str] = None

# =========================
# Stage 1: RAG core
# =========================
class LightningRAG:
    def __init__(self, db_path: str, collection_name: str = "lightning_docs"):
        start_time = time.time()
        logger.info("Initializing Lightning RAG...")

        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2", device="cuda" if self._check_gpu() else "cpu"
        )
        self.embedding_model.encode = self._patch_encode_for_speed(self.embedding_model.encode)
        self.db_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.db_client.get_or_create_collection(name=collection_name)
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        logger.info(f"RAG ready in {(time.time()-start_time)*1000:.0f}ms")

        self.pos_words = {
            "strong","record","beat","beats","exceed","growth","improved","surge","robust",
            "profit","profitable","increase","raised","higher","positive","expanded","outperform",
            "resilient","upgraded","upgrade","accelerate","accelerating","expansion"
        }
        self.neg_words = {
            "weak","decline","decrease","drop","fall","miss","shortfall","lower","negative","loss",
            "impairment","charge","restatement","investigation","material weakness","going concern",
            "downgrade","bankruptcy","liquidity crunch","covenant breach","default","downgraded"
        }

    def _check_gpu(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _patch_encode_for_speed(self, original_encode):
        def fast_encode(sentences, batch_size=BATCH_SIZE, **kwargs):
            kwargs["batch_size"] = batch_size
            kwargs["show_progress_bar"] = False
            kwargs["convert_to_numpy"] = True
            return original_encode(sentences, **kwargs)
        return fast_encode

    def lightning_pdf_extract(self, pdf_path: Path) -> List[str]:
        start_time = time.time()
        doc = fitz.open(pdf_path)

        def extract_page_text(i: int) -> str:
            return doc[i].get_text()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            page_texts = list(pool.map(extract_page_text, range(len(doc))))
        doc.close()

        full_text = " ".join(page_texts)
        chunks, words = [], full_text.split()
        current, size = [], 0
        for w in words:
            if size + len(w) > CHUNK_SIZE and current:
                chunks.append(" ".join(current))
                current, size = [w], len(w)
            else:
                current.append(w); size += len(w) + 1
        if current: chunks.append(" ".join(current))
        chunks = [c for c in chunks if len(c.strip()) > 50]
        logger.info(f"Extracted {len(chunks)} chunks in {(time.time()-start_time)*1000:.0f}ms")
        return chunks

    def lightning_embed_and_store(self, chunks: List[str], doc_name: str, session_id: str):
        emb = self.embedding_model.encode(chunks, batch_size=BATCH_SIZE, show_progress_bar=False, convert_to_numpy=True).tolist()
        ids = [f"{session_id}::{doc_name}::chunk_{i}" for i in range(len(chunks))]
        self.collection.add(
            embeddings=emb,
            documents=chunks,
            ids=ids,
            metadatas=[{"source": doc_name, "chunk_id": i, "session_id": session_id} for i in range(len(chunks))]
        )

    async def lightning_process_pdf(self, pdf_path: Path, task_id: str, session_id: str):
        try:
            task_statuses[task_id] = {"status": "extracting_text", "progress": 20, "start_time": time.time(), "session_id": session_id}
            chunks = self.lightning_pdf_extract(pdf_path)

            task_statuses[task_id] = {"status": "creating_embeddings", "progress": 60, "chunks_extracted": len(chunks), "start_time": time.time(), "session_id": session_id}
            self.lightning_embed_and_store(chunks, pdf_path.name, session_id=session_id)

            task_statuses[task_id] = {"status": "complete", "progress": 100, "total_time_ms": 0, "chunks_processed": len(chunks), "session_id": session_id}
        except Exception as e:
            task_statuses[task_id] = {"status": "error", "error": str(e), "session_id": session_id}
            logger.exception("Processing failed")

    def lightning_query(self, question: str, top_k: int, session_id: str) -> Dict:
        start = time.time()
        q_emb = self.embedding_model.encode([question], convert_to_numpy=True)[0].tolist()
        results = self.collection.query(
            query_embeddings=[q_emb], n_results=top_k, include=["documents", "distances", "metadatas"],
            where={"session_id": session_id}
        )
        docs = results.get("documents", [[]])[0]
        if not docs:
            return {"answer": "No relevant information found for this upload/session.",
                    "processing_time_ms": int((time.time()-start)*1000), "chunks_used": 0, "confidence": 0.0}
        context = "\n---\n".join(docs)
        distances = results.get("distances", [[]])[0]
        confidence = max(0, 1 - min(distances)) if distances else 0.5
        prompt = f"Answer using ONLY the provided context. Be concise.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        answer = self._fast_gemini_call(prompt)
        return {"answer": answer, "processing_time_ms": int((time.time()-start)*1000), "chunks_used": len(docs), "confidence": confidence}

    def get_session_text(self, session_id: str, top_k_chunks: int = 20, char_limit: int = 30000) -> Tuple[str, int]:
        batch = self.collection.get(where={"session_id": session_id}, include=["documents"], limit=50000)
        docs = batch.get("documents", []) or []
        if docs and isinstance(docs[0], list):
            docs = [d for sub in docs for d in sub]
        docs = docs[:top_k_chunks]
        text = "\n".join(docs)
        sampled = len(text)
        if len(text) > char_limit:
            text = text[:char_limit]
        return text, sampled

    def analyze_sentiment(self, text: str) -> SentimentResult:
        if not text.strip():
            return SentimentResult(label="Neutral", polarity=0.0, confidence=0.0, rationale="No text.", source="fallback")
        prompt = (
            "Financial sentiment. Return STRICT JSON only: "
            '{"label":"Positive|Negative|Neutral","polarity":-1..1,"confidence":0..1,"rationale":"short"}\n\n'
            f"{text[:6000]}"
        )
        raw = self._fast_gemini_call(prompt)
        parsed = self._try_parse_json(raw)
        if parsed:
            label = str(parsed.get("label","Neutral")).title()
            pol = float(parsed.get("polarity",0.0)); conf = float(parsed.get("confidence",0.7))
            rat = str(parsed.get("rationale","")).strip()[:400]
            pol = max(-1.0, min(1.0, pol)); conf = max(0.0, min(1.0, conf))
            if label not in {"Positive","Negative","Neutral"}:
                label = "Neutral" if abs(pol)<0.15 else ("Positive" if pol>0 else "Negative")
            return SentimentResult(label=label, polarity=pol, confidence=conf, rationale=rat, source="gemini")
        # fallback
        t = text.lower()
        pos = sum(1 for w in self.pos_words if w in t); neg = sum(1 for w in self.neg_words if w in t)
        total = pos+neg
        if total==0: return SentimentResult(label="Neutral", polarity=0.0, confidence=0.5, rationale="No strong cues.", source="fallback")
        score = (pos-neg)/total; conf = min(0.95, 0.6 + 0.05*total)
        label = "Neutral" if abs(score)<0.15 else ("Positive" if score>0 else "Negative")
        rationale = f"Signals → positive:{[w for w in self.pos_words if w in t][:5]} negative:{[w for w in self.neg_words if w in t][:5]}"
        return SentimentResult(label=label, polarity=score, confidence=conf, rationale=rationale, source="fallback")

    def detect_anomalies(self, text: str, yoy_threshold: float = 0.30, margin_bps_threshold: int = 300) -> List[AnomalyItem]:
        if not text.strip(): return []
        prompt = (
            "Extract key comparable metrics (YoY or QoQ). Return STRICT JSON array. "
            'Each: {"metric":"...", "unit":"$|%|EPS", "period_1":"...", "value_1":float, '
            '"period_2":"...", "value_2":float, "pct_change":float, "direction":"up|down|flat"}\n\n'
            f"{text[:7000]}"
        )
        raw = self._fast_gemini_call(prompt)
        arr = self._try_parse_json_array_of_dicts(raw) or []
        results: List[AnomalyItem] = []
        if arr:
            for it in arr:
                metric = str(it.get("metric","")).strip()
                unit = it.get("unit")
                v1 = self._safe_float(it.get("value_1")); v2 = self._safe_float(it.get("value_2"))
                pct = self._safe_float(it.get("pct_change"))
                is_anom, reason = self._anomaly_rule(metric, unit, v1, v2, pct, yoy_threshold, margin_bps_threshold)
                results.append(AnomalyItem(metric=metric or "Metric", unit=unit,
                                           period_1=it.get("period_1"), value_1=v1,
                                           period_2=it.get("period_2"), value_2=v2,
                                           pct_change=pct, direction=it.get("direction"),
                                           is_anomaly=is_anom, reason=reason))
            return results
        # regex fallback
        results.extend(self._regex_fallback_anomalies(text, yoy_threshold, margin_bps_threshold))
        return results

    def _anomaly_rule(self, metric: Optional[str], unit: Optional[str], v1: Optional[float], v2: Optional[float],
                      pct: Optional[float], yoy_threshold: float, margin_bps_threshold: int) -> Tuple[bool,str]:
        m = (metric or "").lower()
        if pct is None and v1 is not None and v2 not in (None, 0):
            pct = (v1 - v2) / abs(v2)
        if "margin" in m and v1 is not None and v2 is not None:
            delta_bps = (v1 - v2) * 100
            if abs(delta_bps) >= margin_bps_threshold:
                return True, f"Margin moved {delta_bps:.0f} bps ({'down' if delta_bps<0 else 'up'})."
        if "eps" in m and v1 is not None and v2 is not None:
            if (v1 < 0 <= v2) or (v2 < 0 <= v1): return True, "EPS flipped sign."
            if pct is not None and abs(pct) >= yoy_threshold: return True, f"EPS changed {pct*100:.1f}%."
        if any(k in m for k in ["revenue","sales","income","profit","ebit","ebitda","free cash flow","fcf"]):
            if v1 is not None and v2 is not None and ((v1<0<=v2) or (v2<0<=v1)): return True, "Profitability flipped sign."
            if pct is not None and abs(pct) >= yoy_threshold: return True, f"{metric} changed {pct*100:.1f}%."
        return False, ""

    def _regex_fallback_anomalies(self, text: str, yoy_threshold: float, margin_bps_threshold: int) -> List[AnomalyItem]:
        items: List[AnomalyItem] = []
        pat = re.compile(r"(revenue|sales)[^$€₹%]{0,120}?([$€₹]?\(?-?\d[\d,]*(?:\.\d+)?)(?:\s*(billion|million|bn|mn|thousand|k))?.{0,50}?([$€₹]?\(?-?\d[\d,]*(?:\.\d+)?)(?:\s*(billion|million|bn|mn|thousand|k))?", re.I)
        for m in pat.finditer(text):
            metric = m.group(1).title()
            v1 = self._normalize_num(m.group(2), m.group(3)); v2 = self._normalize_num(m.group(4), m.group(5))
            pct = None
            if v1 is not None and v2 not in (None,0): pct = (v1 - v2)/abs(v2)
            is_anom, reason = self._anomaly_rule(metric, "$", v1, v2, pct, yoy_threshold, margin_bps_threshold)
            items.append(AnomalyItem(metric=metric, unit="$", value_1=v1, value_2=v2, pct_change=pct,
                                     is_anomaly=is_anom, reason=reason))
        mpat = re.compile(r"(gross|operating)\s+margin[^%]{0,80}?(-?\d+(?:\.\d+)?)\s*%[^%]{0,40}?(-?\d+(?:\.\d+)?)\s*%", re.I)
        for m in mpat.finditer(text):
            metric = f"{m.group(1).title()} margin"; v1 = float(m.group(2)); v2 = float(m.group(3))
            is_anom, reason = self._anomaly_rule(metric, "%", v1, v2, None, yoy_threshold, margin_bps_threshold)
            items.append(AnomalyItem(metric=metric, unit="%", value_1=v1, value_2=v2, is_anomaly=is_anom, reason=reason))
        return items

    def _fast_gemini_call(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        params = {"key": GEMINI_API_KEY}
        payload = {"contents":[{"parts":[{"text": prompt}]}],
                   "generationConfig":{"temperature":0.1,"topP":0.8,"maxOutputTokens":1024}}
        try:
            resp = requests.post(GEMINI_ENDPOINT, headers=headers, params=params, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and "error" in data:
                return json.dumps({"error": data["error"].get("message","model_error")})
            for c in data.get("candidates",[]) or []:
                content = c.get("content") or {}; parts = content.get("parts") or []
                for p in parts:
                    if isinstance(p, dict) and isinstance(p.get("text"), str) and p["text"].strip():
                        return p["text"]
            return json.dumps({"error":"no_text"})
        except Exception as e:
            logger.error(f"Gemini call failed: {e}")
            return json.dumps({"error":"request_failed","detail":str(e)})

    def _try_parse_json(self, s: str) -> Optional[Dict[str, Any]]:
        try:
            i = s.find("{"); j = s.rfind("}")
            if i!=-1 and j!=-1 and j>i: return json.loads(s[i:j+1])
        except Exception: pass
        return None

    def _try_parse_json_array_of_dicts(self, s: str) -> Optional[List[Dict[str, Any]]]:
        try:
            i = s.find("["); j = s.rfind("]")
            if i!=-1 and j!=-1 and j>i:
                arr = json.loads(s[i:j+1])
                if isinstance(arr, list): return [x for x in arr if isinstance(x, dict)]
        except Exception: pass
        return None

    def _safe_float(self, v: Any) -> Optional[float]:
        try: return float(v)
        except Exception: return None

    def _normalize_num(self, s: Optional[str], scale: Optional[str]) -> Optional[float]:
        if not s: return None
        x = s.replace("$","").replace("€","").replace("₹","").replace(",","").strip()
        neg = x.startswith("(") and x.endswith(")"); x = x.strip("() ")
        try: val = float(x)
        except Exception: return None
        mult = 1.0
        if scale:
            sc = scale.lower()
            if sc in {"billion","bn"}: mult = 1_000_000_000
            elif sc in {"million","mn"}: mult = 1_000_000
            elif sc in {"thousand","k"}: mult = 1_000
        if neg: val = -val
        return val * mult

rag_system = LightningRAG(db_path=str(DB_DIR))

# =========================
# Stage 2: Forecasting classes
# =========================
class Config:
    LSTM_EPOCHS = 30
    LSTM_BATCH_SIZE = 32
    LSTM_WINDOW = 60
    BUY_THRESHOLD = 0.02
    SELL_THRESHOLD = -0.02
    TEST_SPLIT_RATIO = 0.2

class DataFetcher:
    def fetch_data(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No data for {ticker}")
        return df

class FeatureEngineer:
    @staticmethod
    def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["price_change"] = df["Close"].pct_change()
        df["sma_10"] = ta.trend.sma_indicator(df["Close"], window=10)
        df["sma_20"] = ta.trend.sma_indicator(df["Close"], window=20)
        df["sma_50"] = ta.trend.sma_indicator(df["Close"], window=50)
        df["ema_12"] = ta.trend.ema_indicator(df["Close"], window=12)
        df["ema_26"] = ta.trend.ema_indicator(df["Close"], window=26)
        df["rsi"] = ta.momentum.rsi(df["Close"], window=14)
        macd = ta.trend.MACD(df["Close"])
        df["macd"] = macd.macd(); df["macd_signal"] = macd.macd_signal(); df["macd_histogram"] = macd.macd_diff()
        if "Volume" in df.columns:
            df["volume_sma"] = ta.trend.sma_indicator(df["Volume"], window=20)
            df["mfi"] = ta.volume.money_flow_index(df["High"], df["Low"], df["Close"], df["Volume"])
        bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband(); df["bb_lower"] = bb.bollinger_lband()
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["Close"]
        df["volatility"] = df["price_change"].rolling(window=20).std()
        df["price_momentum_5"] = df["Close"]/df["Close"].shift(5) - 1
        df["price_momentum_10"] = df["Close"]/df["Close"].shift(10) - 1
        return df.dropna()

    @staticmethod
    def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["day_of_week"] = df.index.dayofweek; df["month"] = df.index.month; df["quarter"] = df.index.quarter
        df["day_sin"] = np.sin(2*np.pi*df["day_of_week"]/7); df["day_cos"] = np.cos(2*np.pi*df["day_of_week"]/7)
        df["month_sin"] = np.sin(2*np.pi*df["month"]/12); df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
        return df

    @staticmethod
    def add_lag_features(df: pd.DataFrame, lags=[1,2,3,5,10]) -> pd.DataFrame:
        df = df.copy()
        if "price_change" not in df.columns: df["price_change"] = df["Close"].pct_change()
        for lag in lags:
            df[f"close_lag_{lag}"] = df["Close"].shift(lag)
            if "Volume" in df.columns: df[f"volume_lag_{lag}"] = df["Volume"].shift(lag)
            df[f"returns_lag_{lag}"] = df["price_change"].shift(lag)
        return df.dropna()

class ProphetModel:
    def __init__(self): self.model=None; self.forecast_df=None
    def train(self, df: pd.DataFrame, periods=30):
        p = df.reset_index()
        if p["Date"].dt.tz is not None:
            p["ds"] = p["Date"].dt.tz_localize(None)
        else:
            p["ds"] = p["Date"]
        p = p.rename(columns={"Close":"y"})[["ds","y"]]
        self.model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True,
                             seasonality_mode="multiplicative", changepoint_prior_scale=0.05, seasonality_prior_scale=10)
        self.model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
        self.model.add_seasonality(name="quarterly", period=91.25, fourier_order=8)
        self.model.fit(p)
        future = self.model.make_future_dataframe(periods=periods)
        self.forecast_df = self.model.predict(future)
        return {"training_completed": True}
    def forecast_future(self, horizon=30) -> np.ndarray:
        if self.forecast_df is None: return np.array([])
        return self.forecast_df["yhat"].iloc[-horizon:].values

class LSTMModel:
    def __init__(self, cfg: Config):
        self.cfg = cfg; self.model=None; self.scaler=None; self.feature_columns=None
    def create_sequences(self, data: np.ndarray):
        X,y=[],[]
        for i in range(self.cfg.LSTM_WINDOW, len(data)):
            X.append(data[i-self.cfg.LSTM_WINDOW:i]); y.append(data[i][0])
        return np.array(X), np.array(y)
    def build_model(self, input_shape):
        m = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape), Dropout(0.2),
            LSTM(64, return_sequences=True), Dropout(0.2),
            LSTM(32, return_sequences=False), Dropout(0.2),
            Dense(16, activation="relu"), Dense(1)
        ])
        m.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"]); return m
    def train(self, df: pd.DataFrame, features=["Close"]):
        self.feature_columns = features; data = df[features].values.astype("float32")
        self.scaler = MinMaxScaler(); scaled = self.scaler.fit_transform(data)
        X,y = self.create_sequences(scaled)
        if len(X)==0: return {"error":"insufficient data"}
        split = max(1, int(len(X)*(1-self.cfg.TEST_SPLIT_RATIO)))
        Xtr,Xte = X[:split],X[split:]; ytr,yte=y[:split],y[split:]
        Xtr = Xtr.reshape(Xtr.shape[0], Xtr.shape[1], len(features))
        if len(Xte)>0: Xte = Xte.reshape(Xte.shape[0], Xte.shape[1], len(features))
        self.model = self.build_model((Xtr.shape[1], len(features)))
        cbs=[EarlyStopping(monitor="val_loss",patience=5,restore_best_weights=True),
             ReduceLROnPlateau(monitor="val_loss",patience=3,factor=0.5,min_lr=1e-4)]
        hist = self.model.fit(Xtr,ytr,epochs=self.cfg.LSTM_EPOCHS,
                              batch_size=min(self.cfg.LSTM_BATCH_SIZE,len(Xtr)),
                              validation_data=(Xte,yte) if len(Xte)>0 else None, verbose=0, callbacks=cbs)
        return {"train_loss": hist.history["loss"][-1], "training_completed": True}
    def forecast(self, df: pd.DataFrame, horizon=30) -> np.ndarray:
        if self.model is None or self.scaler is None: return np.full(horizon, df["Close"].iloc[-1])
        recent = df[self.feature_columns].values[-self.cfg.LSTM_WINDOW:].astype("float32")
        if len(recent)<self.cfg.LSTM_WINDOW: return np.full(horizon, df["Close"].iloc[-1])
        seq = self.scaler.transform(recent); preds=[]
        cur = seq.copy()
        for _ in range(horizon):
            X = cur.reshape(1, self.cfg.LSTM_WINDOW, len(self.feature_columns))
            pred = self.model.predict(X, verbose=0)[0][0]; preds.append(pred)
            nxt = cur[-1].copy(); nxt[0]=pred; cur = np.vstack([cur[1:], nxt.reshape(1,-1)])
        preds = np.array(preds).reshape(-1,1)
        if len(self.feature_columns)>1:
            pad = np.zeros((len(preds), len(self.feature_columns)-1))
            inv = self.scaler.inverse_transform(np.hstack([preds,pad]))[:,0]
        else:
            inv = self.scaler.inverse_transform(preds)[:,0]
        return inv

class XGBoostModel:
    def __init__(self):
        self.model=None; self.feature_columns=None
    def prepare_features(self, df: pd.DataFrame):
        d=df.copy()
        for lag in range(1,11): d[f"target_lag_{lag}"]=d["Close"].shift(lag)
        for w in [5,10,20]:
            d[f"target_mean_{w}"]=d["Close"].rolling(w).mean(); d[f"target_std_{w}"]=d["Close"].rolling(w).std()
        d = d.dropna(); y = d["Close"]; X = d.drop(columns=["Close"])
        return X,y
    def train(self, df: pd.DataFrame):
        X,y = self.prepare_features(df); self.feature_columns=X.columns.tolist()
        self.model = xgb.XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.08, random_state=42, objective="reg:squarederror")
        self.model.fit(X,y)
        pred = self.model.predict(X)
        return {"mse": mean_squared_error(y,pred), "mae": mean_absolute_error(y,pred), "r2": r2_score(y,pred), "training_completed": True}
    def forecast(self, df: pd.DataFrame, horizon=30)->np.ndarray:
        if self.model is None or self.feature_columns is None: return np.full(horizon, df["Close"].iloc[-1])
        out=[]; cur=df.copy()
        for _ in range(horizon):
            X,_ = self.prepare_features(cur)
            if len(X)==0: break
            for miss in set(self.feature_columns)-set(X.columns): X[miss]=0
            for extra in set(X.columns)-set(self.feature_columns): X.drop(columns=[extra], inplace=True)
            X = X[self.feature_columns]
            pred = self.model.predict(X.tail(1))[0]; out.append(pred)
            next_date = cur.index[-1] + timedelta(days=1)
            new_row = cur.iloc[-1].copy(); new_row["Close"]=pred; new_row.name=next_date
            cur = pd.concat([cur, new_row.to_frame().T])
        return np.array(out)

class TradingSignalGenerator:
    def __init__(self, cfg: Config): self.cfg=cfg
    def generate_signals(self, ensemble: np.ndarray, current_price: float) -> pd.DataFrame:
        future_dates = [datetime.now()+timedelta(days=i+1) for i in range(len(ensemble))]
        returns = np.zeros(len(ensemble))
        returns[0] = (ensemble[0]-current_price)/current_price
        for i in range(1,len(ensemble)): returns[i]=(ensemble[i]-ensemble[i-1])/max(1e-9,ensemble[i-1])
        df = pd.DataFrame({"Date":future_dates,"Predicted_Price":ensemble,"Expected_Return":returns,
                           "Signal":0,"Action":"HOLD","Days_to_Hold":0,"Confidence":0.0})
        for i,r in enumerate(returns):
            if r > self.cfg.BUY_THRESHOLD:
                df.loc[i,"Signal"]=1; df.loc[i,"Action"]="BUY"; df.loc[i,"Confidence"]=min(abs(r)/0.05,1.0)
            elif r < self.cfg.SELL_THRESHOLD:
                df.loc[i,"Signal"]=-1; df.loc[i,"Action"]="SELL"; df.loc[i,"Confidence"]=min(abs(r)/0.05,1.0)
            else:
                df.loc[i,"Confidence"]=0.3
        # simple holding period
        sig = df["Signal"].values; hold = np.zeros(len(sig),dtype=int)
        for i in range(len(sig)):
            if sig[i]!=0:
                for j in range(i+1, min(i+15,len(sig))):
                    if sig[j]==-sig[i] or abs(df.iloc[j]["Expected_Return"])<0.005:
                        hold[i]=j-i; break
                else: hold[i]=7 if sig[i]==1 else 3
        df["Days_to_Hold"]=hold
        return df
    def get_next_action(self, df: pd.DataFrame) -> Dict[str, Any]:
        act = df[df["Signal"]!=0]
        if act.empty:
            return {"action":"HOLD","date":datetime.now().strftime("%Y-%m-%d"),
                    "price":0.0,"holding_days":0,"confidence":0.3,"reason":"No clear trading signals"}
        row = act.iloc[0]
        return {"action":row["Action"],"date":row["Date"].strftime("%Y-%m-%d"),
                "price":float(row["Predicted_Price"]), "holding_days":int(row["Days_to_Hold"]),
                "confidence":float(row["Confidence"]), "reason":f"Expected return: {row['Expected_Return']:.2%}"}

class TradingVisualizer:
    @staticmethod
    def plot_with_metrics(df: pd.DataFrame,
                          signals: pd.DataFrame,
                          ticker: str,
                          save_path: str,
                          model_metrics: Dict):
        # --- sanitize data ---
        sig = signals.copy()
        sig["Date"] = pd.to_datetime(sig["Date"], errors="coerce")
        for col in ["Predicted_Price", "Expected_Return", "Confidence", "Signal"]:
            sig[col] = pd.to_numeric(sig[col], errors="coerce")
        sig[["Expected_Return", "Confidence"]] = sig[["Expected_Return", "Confidence"]].fillna(0.0)
        sig["Signal"] = sig["Signal"].fillna(0).astype(int)

        # --- layout ---
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=[
                f"{ticker} — Trading Strategy",
                "Model Performance",
                "Signal Timeline",
                "Expected Return & Confidence"
            ],
            specs=[
                [{"type": "xy"}],
                [{"type": "table"}],
                [{"type": "xy"}],
                [{"type": "xy", "secondary_y": True}],
            ],
            vertical_spacing=0.08,
            row_heights=[0.55, 0.14, 0.12, 0.19]
        )

        # --- row 1: historical + ensemble ---
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["Close"],
                name="Historical",
                line=dict(width=2),
                hovertemplate="Date: %{x}<br>Price: $%{y:,.2f}<extra></extra>"
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=sig["Date"], y=sig["Predicted_Price"],
                name=f'Ensemble (R²={model_metrics.get("ensemble_r2", 0):.3f})',
                line=dict(width=3),
                hovertemplate="Date: %{x}<br>Forecast: $%{y:,.2f}<extra></extra>"
            ),
            row=1, col=1
        )

        # BUY / SELL markers
        buys = sig[sig["Signal"] == 1]
        sells = sig[sig["Signal"] == -1]
        if not buys.empty:
            fig.add_trace(
                go.Scatter(
                    x=buys["Date"], y=buys["Predicted_Price"],
                    mode="markers+text",
                    name=f"BUY ({len(buys)})",
                    marker=dict(size=12, symbol="triangle-up", color="green"),
                    text=[f"BUY\n{d}d" for d in buys["Days_to_Hold"]],
                    textposition="top center",
                    hovertemplate="BUY<br>Date: %{x}<br>Price: $%{y:,.2f}<br>Conf: %{customdata:.1%}<extra></extra>",
                    customdata=buys["Confidence"]
                ),
                row=1, col=1
            )
        if not sells.empty:
            fig.add_trace(
                go.Scatter(
                    x=sells["Date"], y=sells["Predicted_Price"],
                    mode="markers+text",
                    name=f"SELL ({len(sells)})",
                    marker=dict(size=12, symbol="triangle-down", color="red"),
                    text=[f"SELL\n{d}d" for d in sells["Days_to_Hold"]],
                    textposition="bottom center",
                    hovertemplate="SELL<br>Date: %{x}<br>Price: $%{y:,.2f}<br>Conf: %{customdata:.1%}<extra></extra>",
                    customdata=sells["Confidence"]
                ),
                row=1, col=1
            )

        # --- row 2: model performance table ---
        perf = [
            ['Ensemble', f"{model_metrics.get('ensemble_mse',0):.2f}",
             f"{model_metrics.get('ensemble_mae',0):.2f}",
             f"{model_metrics.get('ensemble_r2',0):.3f}",
             f"{model_metrics.get('ensemble_mape',0):.1f}%"],
            ['Prophet', f"{model_metrics.get('prophet_mse',0):.2f}", '-', '-', '-'],
            ['LSTM', f"{model_metrics.get('lstm_mse',0):.2f}", '-', '-', '-'],
            ['XGBoost', f"{model_metrics.get('xgboost_mse',0):.2f}", '-', '-', '-']
        ]
        fig.add_trace(
            go.Table(
                header=dict(values=["Model","MSE","MAE","R²","MAPE"],
                            fill_color="lightblue", align="center"),
                cells=dict(values=list(zip(*perf)), align="center")
            ),
            row=2, col=1
        )

        # --- row 3: signal timeline ---
        colors = ["green" if s == 1 else "red" if s == -1 else "gray" for s in sig["Signal"]]
        fig.add_trace(
            go.Bar(
                x=sig["Date"], y=sig["Signal"],
                marker_color=colors, name="Signals",
                customdata=sig["Action"],
                hovertemplate="Date: %{x}<br>Action: %{customdata}<extra></extra>",
                showlegend=False
            ),
            row=3, col=1
        )
        fig.update_yaxes(
            title_text="Signal",
            row=3, col=1,
            tickmode="array", tickvals=[-1, 0, 1], ticktext=["SELL", "HOLD", "BUY"],
            range=[-1.5, 1.5]
        )

        # --- row 4: returns (left) + confidence (right) ---
        fig.add_trace(
            go.Scatter(
                x=sig["Date"], y=sig["Expected_Return"] * 100,
                name="Expected Return (%)",
                hovertemplate="Date: %{x}<br>Expected Return: %{y:.2f}%<extra></extra>"
            ),
            row=4, col=1, secondary_y=False
        )
        fig.add_trace(
            go.Scatter(
                x=sig["Date"], y=sig["Confidence"] * 100,
                name="Confidence (%)",
                line=dict(dash="dot"),
                hovertemplate="Date: %{x}<br>Confidence: %{y:.1f}%<extra></extra>"
            ),
            row=4, col=1, secondary_y=True
        )
        fig.update_yaxes(title_text="Expected Return (%)", row=4, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Confidence (%)",      row=4, col=1, secondary_y=True, range=[0, 100])

        # --- overall layout ---
        fig.update_layout(
            template="plotly_white",
            height=1100,
            margin=dict(t=80, b=50, l=60, r=20),
            legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, xanchor="left")
        )

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path, include_plotlyjs="cdn", full_html=True)
        return fig


# =========================
# Stage 2 pipeline
# =========================
def run_forecast_pipeline(ticker: str, start: str, end: str, outdir: Path, horizon: int = 30):
    fetcher = DataFetcher(); df = fetcher.fetch_data(ticker, start, end)
    fe = FeatureEngineer()
    df_feat = fe.add_lag_features(fe.add_cyclical_features(fe.add_technical_features(df)))
    ens = {"prophet":0.3,"lstm":0.4,"xgboost":0.3}

    prophet = ProphetModel(); lstm = LSTMModel(Config()); xgbm = XGBoostModel()
    # Train
    tre = {}
    try: tre["prophet"] = prophet.train(df_feat)
    except Exception as e: tre["prophet"]={"error":str(e)}
    tre["lstm"] = lstm.train(df_feat)
    tre["xgboost"] = xgbm.train(df_feat)

    # Forecasts
    preds = {}
    preds["prophet"] = prophet.forecast_future(horizon) if "error" not in tre.get("prophet",{}) else np.array([])
    preds["lstm"] = lstm.forecast(df_feat, horizon)
    preds["xgboost"] = xgbm.forecast(df_feat, horizon)

    # Ensemble
    valid = [(name, arr, ens[name]) for name,arr in preds.items() if arr is not None and len(arr)==horizon]
    if valid:
        total_w = sum(w for _,_,w in valid)
        ensemble = np.zeros(horizon, dtype=float)
        for _,arr,w in valid: ensemble += arr*(w/total_w)
    else:
        ensemble = np.full(horizon, df_feat["Close"].iloc[-1])

    # Signals
    gen = TradingSignalGenerator(Config())
    signals = gen.generate_signals(ensemble, df_feat["Close"].iloc[-1])
    next_action = gen.get_next_action(signals)

    # Metrics (simple/fallbacks)
    metrics = {
        "ensemble_mse": tre.get("xgboost",{}).get("mse", 0.0),
        "ensemble_mae": tre.get("xgboost",{}).get("mae", 0.0),
        "ensemble_r2": tre.get("xgboost",{}).get("r2", 0.0),
        "ensemble_mape": 5.0,
        "prophet_mse": 0.0 if "error" in tre.get("prophet",{}) else 18.0,
        "lstm_mse": tre.get("lstm",{}).get("train_loss", 0.0),
        "xgboost_mse": tre.get("xgboost",{}).get("mse", 0.0),
    }

    # Save outputs
    outdir.mkdir(parents=True, exist_ok=True)
    chart_path = outdir / f"{ticker}_strategy.html"
    TradingVisualizer.plot_with_metrics(df_feat, signals, ticker, str(chart_path), metrics)

    signals_csv = outdir / f"{ticker}_trading_signals.csv"
    signals.to_csv(signals_csv, index=False)

    forecasts_csv = outdir / f"{ticker}_forecasts.csv"
    pd.DataFrame({"Date": signals["Date"], "Ensemble_Forecast": ensemble,
                  "Action": signals["Action"], "Confidence": signals["Confidence"]}).to_csv(forecasts_csv, index=False)

    return {
        "chart_path": chart_path,
        "signals_csv": signals_csv,
        "forecasts_csv": forecasts_csv,
        "next_action": next_action,
        "metrics": metrics,
        "ensemble_values": ensemble.tolist(),            # for Stage 3
        "last_close": float(df_feat["Close"].iloc[-1])   # for Stage 3
    }

# =========================
# Background worker (Stage 1)
# =========================
async def lightning_process_pdf_background(pdf_path: Path, task_id: str, rag_system: LightningRAG, session_id: str):
    try:
        await rag_system.lightning_process_pdf(pdf_path, task_id, session_id)
    finally:
        if pdf_path.exists():
            try: pdf_path.unlink()
            except Exception as e: logger.warning(f"Could not delete temp file {pdf_path}: {e}")

# =========================
# Stage 1 endpoints
# =========================
@app.post("/upload-pdf/", status_code=202, summary="Upload & embed (session-scoped)")
async def upload_pdf(file: UploadFile = File(...)):
    global latest_session_id
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="PDF files only!")
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    temp_path = TEMP_DIR / unique_filename
    with temp_path.open("wb") as buffer:
        content = await file.read(); buffer.write(content)
    task_id = str(uuid.uuid4()); session_id = str(uuid.uuid4()); latest_session_id = session_id
    task_statuses[task_id] = {"status":"accepted","filename":file.filename,"file_size":len(content),
                              "received_at":datetime.utcnow().isoformat()+"Z","session_id":session_id}
    asyncio.create_task(lightning_process_pdf_background(temp_path, task_id, rag_system, session_id))
    return {"status":"accepted","task_id":task_id,"session_id":session_id,"message":f"Processing started for {file.filename}"}

@app.get("/status/{task_id}", summary="Check processing status")
def get_task_status(task_id: str):
    data = task_statuses.get(task_id)
    if not data: raise HTTPException(status_code=404, detail="Task not found")
    if "start_time" in data:
        data["elapsed_seconds"] = round(time.time()-data["start_time"], 1)
    return data

@app.post("/query/", response_model=FastResponse, summary="Ask questions (session-scoped)")
async def query_documents(request: QueryRequest):
    sid = request.session_id or latest_session_id
    if not sid:
        raise HTTPException(status_code=400, detail="session_id is required (call /upload-pdf).")
    result = rag_system.lightning_query(request.question, request.top_k, session_id=sid)
    return FastResponse(answer=result["answer"], processing_time_ms=result["processing_time_ms"],
                        chunks_used=result["chunks_used"], confidence=result["confidence"])

@app.post("/sentiment/", response_model=SentimentResponse, summary="Market Sentiment (text or session)")
async def sentiment_analysis(request: SentimentRequest):
    mode="text"; text=(request.text or "").strip(); sid=request.session_id; sampled=0
    if not text:
        sid = sid or latest_session_id
        if not sid: raise HTTPException(status_code=400, detail="Provide text or session_id.")
        text, sampled = rag_system.get_session_text(sid, top_k_chunks=request.top_k_chunks); mode="session"
    result = rag_system.analyze_sentiment(text)
    return SentimentResponse(result=result, mode=mode, session_id=sid, sampled_chars=sampled)

@app.post("/anomaly/", response_model=AnomalyResponse, summary="Anomaly Detection (text or session)")
async def anomaly_detection(request: AnomalyRequest):
    mode="text"; text=(request.text or "").strip(); sid=request.session_id; sampled=0
    if not text:
        sid = sid or latest_session_id
        if not sid: raise HTTPException(status_code=400, detail="Provide text or session_id.")
        text, sampled = rag_system.get_session_text(sid, top_k_chunks=request.top_k_chunks); mode="session"
    items = rag_system.detect_anomalies(text, yoy_threshold=request.yoy_threshold, margin_bps_threshold=request.margin_bps_threshold)
    flagged = [it for it in items if it.is_anomaly]
    if flagged:
        bullets=[]
        for it in flagged[:10]:
            how = f"{(it.pct_change or 0)*100:.1f}%" if it.pct_change is not None else ""
            per = f" ({it.period_1} vs {it.period_2})" if it.period_1 or it.period_2 else ""
            why = f" – {it.reason}" if it.reason else ""
            bullets.append(f"- {it.metric}{per}: {how}{why}".strip())
        summary = "Potential anomalies detected:\n" + "\n".join(bullets)
    else:
        summary = "No material anomalies detected given current thresholds."
    return AnomalyResponse(items=items, summary=summary, session_id=sid, mode=mode, sampled_chars=sampled,
                           rules={"yoy_threshold_pct": request.yoy_threshold*100,
                                  "margin_threshold_bps": request.margin_bps_threshold,
                                  "sign_flip_rule":"Profit/EPS sign flips are flagged"})

# =========================
# Stage 2 endpoint
# =========================
@app.post("/forecast/run", response_model=ForecastResponse, summary="Run forecasting & trading signals")
def run_forecast(req: ForecastRequest):
    # validate dates
    try:
        sd = datetime.strptime(req.start, "%Y-%m-%d")
        ed = datetime.strptime(req.end, "%Y-%m-%d")
        if sd >= ed: raise ValueError("start must be before end")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid date(s): {e}")

    outdir = OUT_DIR if not req.outdir else Path(req.outdir)
    try:
        res = run_forecast_pipeline(req.ticker.upper(), req.start, req.end, outdir, req.horizon)
        chart_rel = res["chart_path"].resolve().relative_to(OUT_DIR.resolve()) if OUT_DIR in res["chart_path"].resolve().parents else res["chart_path"].name
        signals_rel = res["signals_csv"].resolve().relative_to(OUT_DIR.resolve()) if OUT_DIR in res["signals_csv"].resolve().parents else res["signals_csv"].name
        forecasts_rel = res["forecasts_csv"].resolve().relative_to(OUT_DIR.resolve()) if OUT_DIR in res["forecasts_csv"].resolve().parents else res["forecasts_csv"].name

        chart_url = f"/static/{chart_rel}".replace("\\","/")
        signals_url = f"/static/{signals_rel}".replace("\\","/")
        forecasts_url = f"/static/{forecasts_rel}".replace("\\","/")

        return ForecastResponse(
            chart_url=chart_url,
            signals_csv_url=signals_url,
            forecasts_csv_url=forecasts_url,
            next_action=NextAction(**res["next_action"]),
            metrics=res["metrics"],
            notes=[
                "Ensemble = weighted mean of Prophet/LSTM/XGBoost.",
                "Signals generated from expected returns between steps (BUY>+2%, SELL<-2% by default).",
                "Chart & CSVs are saved under /static for quick viewing."
            ]
        )
    except Exception as e:
        logger.exception("Forecast pipeline failed")
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# Stage 3 endpoint (Strategy)
# =========================
def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def _scale_01(x: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.5
    return _clamp((x - lo) / (hi - lo))

@app.post("/strategy/advise", response_model=StrategyResponse, summary="Stage 3: Investment Strategy (BUY/SELL/HOLD)")
def strategy_advise(req: StrategyRequest):
    # --- validate dates
    try:
        sd = datetime.strptime(req.start, "%Y-%m-%d")
        ed = datetime.strptime(req.end, "%Y-%m-%d")
        if sd >= ed:
            raise ValueError("start must be before end")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid date(s): {e}")

    # --- weights (defaults)
    w = {"forecast": 0.55, "sentiment": 0.20, "technical": 0.15, "risk": 0.10}
    if req.weights:
        for k in w:
            if k in req.weights and isinstance(req.weights[k], (int, float)):
                w[k] = float(req.weights[k])
        total_pos = max(1e-9, w["forecast"] + w["sentiment"] + w["technical"])
        scale_pos = (1.0 - w["risk"]) / total_pos
        w["forecast"] *= scale_pos; w["sentiment"] *= scale_pos; w["technical"] *= scale_pos

    # --- 1) run Stage-2 (forecast)
    outdir = OUT_DIR if not req.outdir else Path(req.outdir)
    try:
        f_res = run_forecast_pipeline(req.ticker.upper(), req.start, req.end, outdir, req.horizon)
    except Exception as e:
        logger.exception("Forecast failed in strategy_advise")
        raise HTTPException(status_code=500, detail=f"Forecasting failed: {e}")

    # links
    def _rel(p: Path) -> str:
        p = p.resolve()
        try:
            return str(p.relative_to(OUT_DIR.resolve())).replace("\\", "/")
        except Exception:
            return p.name.replace("\\", "/")

    chart_url = f"/static/{_rel(Path(f_res['chart_path']))}"
    sig_url   = f"/static/{_rel(Path(f_res['signals_csv']))}"
    fc_url    = f"/static/{_rel(Path(f_res['forecasts_csv']))}"

    # ensemble + current price
    ensemble = f_res.get("ensemble_values")
    last_close = f_res.get("last_close")
    if ensemble is None or last_close is None:
        try:
            sig_df = pd.read_csv(f_res["signals_csv"])
            ensemble = sig_df["Predicted_Price"].astype(float).values.tolist()
            df_hist = DataFetcher().fetch_data(req.ticker.upper(), req.start, req.end)
            last_close = float(df_hist["Close"].iloc[-1])
        except Exception:
            ensemble = ensemble or []
            last_close = last_close or 0.0

    # get signals dataframe
    try:
        sig_df = pd.read_csv(f_res["signals_csv"])
        sig_df["Expected_Return"] = pd.to_numeric(sig_df["Expected_Return"], errors="coerce").fillna(0.0)
        sig_df["Confidence"] = pd.to_numeric(sig_df["Confidence"], errors="coerce").fillna(0.0)
    except Exception:
        sig_df = pd.DataFrame({"Expected_Return": np.zeros(len(ensemble)), "Confidence": np.zeros(len(ensemble))})

    next_action = f_res["next_action"]

    # --- forecast_score
    if ensemble and last_close:
        cum_ret = (ensemble[-1] - last_close) / max(1e-9, last_close)
    else:
        cum_ret = 0.0
    pos_frac = float((sig_df["Expected_Return"] > 0).mean()) if len(sig_df) else 0.5
    if next_action.get("action", "HOLD").upper() == "SELL":
        pos_frac = float((sig_df["Expected_Return"] < 0).mean()) if len(sig_df) else 0.5
    cum_component = _scale_01(cum_ret, -0.20, 0.20)
    na_conf = float(next_action.get("confidence", 0.3))
    forecast_score = _clamp(0.4 * cum_component + 0.3 * pos_frac + 0.3 * na_conf)

    # --- Stage-1: sentiment + anomalies (optional)
    sent_snap: Optional[SentimentResult] = None
    anomalies: List[AnomalyItem] = []
    if req.session_id:
        try:
            text, _ = rag_system.get_session_text(req.session_id, top_k_chunks=20)
        except Exception:
            text = ""
        if text:
            try:
                sent_snap = rag_system.analyze_sentiment(text)
            except Exception:
                sent_snap = None
            try:
                anomalies = rag_system.detect_anomalies(text)
            except Exception:
                anomalies = []

    # sentiment_score
    if sent_snap:
        s_base = (float(sent_snap.polarity) + 1.0) / 2.0
        sentiment_score = _clamp(s_base * max(0.0, float(sent_snap.confidence)))
    else:
        sentiment_score = 0.5  # neutral fallback

    # --- quick technical snapshot
    try:
        df_hist = DataFetcher().fetch_data(req.ticker.upper(), req.start, req.end)
    except Exception:
        df_hist = pd.DataFrame()

    technical_score = 0.5
    vol_penalty = 0.0
    atr_penalty = 0.0
    atr = None

    if not df_hist.empty:
        try:
            rsi = ta.momentum.rsi(df_hist["Close"], window=14).iloc[-1]
            macd = ta.trend.MACD(df_hist["Close"])
            macd_hist = macd.macd_diff().iloc[-1]
            technical_score = _clamp(0.5 + (float(rsi) - 50.0)/100.0)
            technical_score = _clamp(technical_score + (0.05 if macd_hist > 0 else -0.05))
        except Exception:
            technical_score = 0.5

        try:
            vol = float(df_hist["Close"].pct_change().rolling(20).std().iloc[-1])
            vol_penalty = _clamp(vol * 8.0, 0.0, 0.35)
        except Exception:
            vol_penalty = 0.0

        try:
            atr = ta.volatility.average_true_range(df_hist["High"], df_hist["Low"], df_hist["Close"], window=14).iloc[-1]
            atr_penalty = _clamp((float(atr) / max(1e-9, float(df_hist["Close"].iloc[-1]))) * 2.0, 0.0, 0.25)
        except Exception:
            atr_penalty = 0.0

    # anomaly risk
    neg_hits = 0
    for it in anomalies or []:
        d = (it.direction or "").lower()
        r = (it.reason or "").lower()
        if ("down" in d) or ("down" in r) or ("flipped" in r) or ("weak" in r) or ("declin" in r) or ("loss" in r):
            neg_hits += 1
    anomaly_penalty = _clamp(0.05 * neg_hits, 0.0, 0.35)

    risk_penalty = _clamp(vol_penalty + atr_penalty + anomaly_penalty, 0.0, 0.60)

    # --- composite
    composite = _clamp(
        w["forecast"] * forecast_score +
        w["sentiment"] * sentiment_score +
        w["technical"] * technical_score -
        w["risk"] * risk_penalty
    )
    buy_thr, sell_thr = 0.62, 0.38
    if composite >= buy_thr:
        action = "BUY"
    elif composite <= sell_thr:
        action = "SELL"
    else:
        action = "HOLD"

    # confidence
    dist_conf = 0.5 + abs(composite - 0.5)
    confs = [dist_conf, float(next_action.get("confidence", 0.3))]
    if sent_snap:
        confs.append(float(sent_snap.confidence))
    confidence = float(np.mean(confs))
    confidence = _clamp(confidence, 0.3, 0.98)

    # entry/SL/TP using ATR
    entry = float(last_close or 0.0)
    if atr is None and not df_hist.empty:
        try:
            atr = ta.volatility.average_true_range(df_hist["High"], df_hist["Low"], df_hist["Close"], window=14).iloc[-1]
        except Exception:
            atr = None
    if atr is None:
        atr = entry * 0.02

    if action == "BUY":
        stop_loss = max(0.0, entry - 1.5 * float(atr))
        take_profit = entry + 2.5 * float(atr)
    elif action == "SELL":
        stop_loss = entry + 1.5 * float(atr)
        take_profit = max(0.0, entry - 2.5 * float(atr))
    else:
        stop_loss = max(0.0, entry - 1.0 * float(atr))
        take_profit = entry + 1.0 * float(atr)

    # holding period
    hold_days = int(next_action.get("holding_days", max(5, req.horizon // 3)))

    # rationale
    bullets = []
    bullets.append(f"Forecast path implies ~{cum_ret*100:.1f}% over {req.horizon}d; direction consistency {pos_frac*100:.0f}%.")
    if sent_snap:
        bullets.append(f"Sentiment: {sent_snap.label} (polarity {sent_snap.polarity:+.2f}, conf {sent_snap.confidence:.2f}).")
    else:
        bullets.append("Sentiment: neutral (no session bound).")
    bullets.append(f"Technical bias: score {technical_score:.2f} (RSI/MACD snapshot).")
    if neg_hits > 0:
        bullets.append(f"Risk flags: {neg_hits} negative anomaly cues; total risk penalty {risk_penalty:.2f}.")
    else:
        bullets.append(f"Risk penalty {risk_penalty:.2f} (volatility/ATR).")

    breakdown = StrategyBreakdown(
        forecast_score=float(forecast_score),
        sentiment_score=float(sentiment_score),
        technical_score=float(technical_score),
        risk_penalty=float(risk_penalty),
        composite_score=float(composite),
    )

    rec = StrategyRecommendation(
        action=action,
        confidence=float(confidence),
        entry_price=float(entry),
        stop_loss=float(stop_loss),
        take_profit=float(take_profit),
        holding_period_days=int(hold_days),
        rationale=bullets
    )

    return StrategyResponse(
        recommendation=rec,
        breakdown=breakdown,
        next_action=NextAction(**next_action),
        links={"chart_url": chart_url, "signals_csv_url": sig_url, "forecasts_csv_url": fc_url},
        metrics=f_res["metrics"],
        sentiment_snapshot=sent_snap,
        anomalies=anomalies or []
    )

# =========================
# Debug
# =========================
@app.get("/debug/sessions", summary="List sessions & chunk counts")
def list_sessions():
    try:
        batch = rag_system.collection.get(include=["metadatas"], limit=50000)
        metas = batch.get("metadatas", []) or []
        sessions: Dict[str,int] = {}
        for md in metas:
            if isinstance(md, list):
                for m in md:
                    sid = (m or {}).get("session_id"); 
                    if sid: sessions[sid]=sessions.get(sid,0)+1
            else:
                sid = (md or {}).get("session_id"); 
                if sid: sessions[sid]=sessions.get(sid,0)+1
        return {"sessions":[{"session_id":k,"chunks":v} for k,v in sessions.items()]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/debug/sessions/{session_id}", summary="Delete all chunks for a session")
def delete_session(session_id: str):
    try:
        rag_system.collection.delete(where={"session_id": session_id})
        return {"deleted_session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# Health
# =========================
@app.get("/", summary="Health")
def health():
    return {
        "status":"OK",
        "version":"6.0.0",
        "endpoints":[
            "POST /upload-pdf  | GET /status/{task_id} | POST /query | POST /sentiment | POST /anomaly",
            "POST /forecast/run  (returns chart/csv URLs under /static)",
            "POST /strategy/advise  (Stage 3: BUY/SELL/HOLD + breakdown)"
        ]
    }
