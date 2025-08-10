# frontend.py
# Streamlit frontend for AkashX FinDocGPT — Stage 1 (RAG Q&A + Sentiment + Anomaly),
# Stage 2 (Forecasting), and Stage 3 (Strategy). Works with your FastAPI backend.
# - Unique keys everywhere (no DuplicateElementId)
# - Strategy tab will use /strategy/recommend if available; otherwise, it falls back
#   to a local aggregator that combines Sentiment + Anomalies + Forecast next_action.

import streamlit as st
import requests
import time
from typing import Optional, Tuple, Dict, Any, List
import pandas as pd
from datetime import date, timedelta

# ============================== Config ==============================
st.set_page_config(page_title="FinDocGPT — Stage 1 • 2 • 3", page_icon="📊", layout="wide")
st.title("📊 FinDocGPT — Stage 1 • Stage 2 • Stage 3")
st.caption("Upload → Embed → Ask → Sentiment/Anomaly → Forecast → Strategy")

DEFAULT_API = "http://localhost:8000"

# ============================== Session state ==============================
if "api_base" not in st.session_state:
    st.session_state.api_base = DEFAULT_API
if "sessions" not in st.session_state:
    st.session_state.sessions: List[Dict[str, Any]] = []
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id: Optional[str] = None

# ============================== Helpers ==============================
def join_url(base: str, path: str) -> str:
    if path.startswith("http://") or path.startswith("https://"):
        return path
    base = base.rstrip("/")
    if path.startswith("/"):
        return base + path
    return base + "/" + path

def api_get(path: str, timeout: int = 30) -> requests.Response:
    return requests.get(join_url(st.session_state.api_base, path), timeout=timeout)

def api_post(path: str, payload: dict = None, files=None, timeout: int = 120) -> requests.Response:
    headers = {} if files else {"Content-Type": "application/json"}
    return requests.post(
        join_url(st.session_state.api_base, path),
        json=payload if files is None else None,
        files=files,
        headers=headers,
        timeout=timeout
    )

def check_api() -> Tuple[bool, str, dict]:
    try:
        r = api_get("/", timeout=15)
        if r.status_code == 200:
            data = {}
            try:
                data = r.json()
            except Exception:
                pass
            return True, "Connected", data
        return False, f"HTTP {r.status_code}", {}
    except requests.exceptions.RequestException as e:
        return False, str(e), {}

def upload_pdf(file) -> Tuple[bool, str, Optional[str], Optional[str]]:
    try:
        files = {"file": (file.name, file, "application/pdf")}
        r = api_post("/upload-pdf/", files=files, timeout=300)
        if r.status_code in (200, 202):
            data = r.json()
            return True, data.get("message", "Accepted"), data.get("session_id"), data.get("task_id")
        return False, r.json().get("detail", f"HTTP {r.status_code}"), None, None
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {e}", None, None

def poll_status(task_id: str, max_wait_s: int = 180) -> Tuple[bool, str]:
    start = time.time()
    ph = st.empty()
    while time.time() - start < max_wait_s:
        try:
            r = api_get(f"/status/{task_id}", timeout=15)
            if r.status_code == 200:
                data = r.json()
                stt = data.get("status")
                prog = data.get("progress", 0)
                ph.info(f"Status: **{stt}** | Progress: **{prog}%**")
                if stt == "complete":
                    ph.success("✅ Processing complete.")
                    return True, "complete"
                if stt == "error":
                    ph.error(f"❌ {data.get('error', 'Unknown error')}")
                    return False, data.get("error", "error")
            else:
                ph.warning(f"Status HTTP {r.status_code}")
        except requests.exceptions.RequestException as e:
            ph.warning(f"Status check error: {e}")
        time.sleep(1.2)
    ph.warning("⏱️ Timed out waiting for completion.")
    return False, "timeout"

def do_query(question: str, session_id: Optional[str], top_k: int = 8):
    payload = {"question": question, "top_k": top_k}
    if session_id:
        payload["session_id"] = session_id
    return api_post("/query/", payload=payload, timeout=120)

def do_sentiment(text: Optional[str], session_id: Optional[str], top_k_chunks: int = 12):
    payload: Dict[str, Any] = {"top_k_chunks": top_k_chunks}
    if text and text.strip():
        payload["text"] = text.strip()
    elif session_id:
        payload["session_id"] = session_id
    return api_post("/sentiment/", payload=payload, timeout=90)

def do_anomaly(text: Optional[str], session_id: Optional[str], top_k_chunks: int = 20,
               yoy_threshold: float = 0.30, margin_bps_threshold: int = 300):
    payload: Dict[str, Any] = {"top_k_chunks": top_k_chunks,
                               "yoy_threshold": yoy_threshold,
                               "margin_bps_threshold": margin_bps_threshold}
    if text and text.strip():
        payload["text"] = text.strip()
    elif session_id:
        payload["session_id"] = session_id
    return api_post("/anomaly/", payload=payload, timeout=120)

def do_forecast(ticker: str, start: str, end: str, horizon: int = 30, outdir: str = "outputs"):
    payload = {"ticker": ticker, "start": start, "end": end, "horizon": horizon, "outdir": outdir}
    return api_post("/forecast/run", payload=payload, timeout=300)

def try_strategy_backend(payload: dict):
    """Attempt to call /strategy/recommend; return (ok, json or error string)."""
    try:
        r = api_post("/strategy/recommend", payload=payload, timeout=120)
        if r.status_code == 200:
            return True, r.json()
        return False, r.json().get("detail", f"HTTP {r.status_code}")
    except Exception as e:
        return False, str(e)

def refresh_remote_sessions_sidebar():
    try:
        r = api_get("/debug/sessions", timeout=20)
        if r.status_code == 200:
            data = r.json()
            remote = data.get("sessions", [])
            if remote:
                st.sidebar.write("**Server Sessions** (chunks):")
                for s in remote[:20]:
                    st.sidebar.caption(f"{s['session_id']} — {s['chunks']}")
    except Exception:
        pass

# ---------- Stage 3 local fallback ----------
def local_strategy_fallback(
    forecast_json: dict,
    sentiment_json: Optional[dict],
    anomalies_json: Optional[dict],
    buy_threshold: float,
    sell_threshold: float
) -> dict:
    """
    Very simple rule-based strategy:
    - Start from forecast next_action (BUY/SELL/HOLD + confidence).
    - Add sentiment polarity (+/-) as a tilt.
    - Penalize if anomalies flagged.
    """
    # Forecast anchor
    na = (forecast_json or {}).get("next_action", {}) if forecast_json else {}
    action = str(na.get("action", "HOLD")).upper()
    conf = float(na.get("confidence", 0.3))

    # Sentiment
    sent_polarity = 0.0
    sent_label = None
    if sentiment_json:
        res = sentiment_json.get("result", {})
        sent_label = res.get("label", "Neutral")
        try:
            sent_polarity = float(res.get("polarity", 0.0))
        except Exception:
            sent_polarity = 0.0

    # Anomalies
    anom_penalty = 0.0
    anom_count = 0
    if anomalies_json:
        items = anomalies_json.get("items", [])
        anom_count = sum(1 for it in items if it.get("is_anomaly"))
        if anom_count >= 3:
            anom_penalty = 0.25
        elif anom_count == 2:
            anom_penalty = 0.15
        elif anom_count == 1:
            anom_penalty = 0.08

    # Score building
    base = 0.0
    if action == "BUY":
        base = 0.35 + 0.4 * conf
    elif action == "SELL":
        base = -0.35 - 0.4 * conf
    else:
        base = 0.0

    # Sentiment tilt: +/- up to ~0.25
    base += max(-0.25, min(0.25, sent_polarity * 0.25))
    # Anomaly penalty
    base -= anom_penalty

    # Decision
    final_action = "HOLD"
    if base >= buy_threshold:
        final_action = "BUY"
    elif base <= -abs(sell_threshold):
        final_action = "SELL"

    reason_lines = []
    reason_lines.append(f"Forecast suggests **{action}** (conf {conf:.0%}).")
    if sent_label:
        reason_lines.append(f"Sentiment: **{sent_label}** (polarity {sent_polarity:+.2f}).")
    if anom_count:
        reason_lines.append(f"Anomalies flagged: **{anom_count}** (penalty {anom_penalty:.2f}).")
    reason_lines.append(f"Composite score: **{base:+.2f}** (BUY≥{buy_threshold:.2f}, SELL≤{-abs(sell_threshold):.2f})")

    return {
        "decision": final_action,
        "score": round(base, 3),
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
        "reasons": reason_lines,
        "used_backend": False,
        "upstream_next_action": na
    }

# ============================== Sidebar ==============================
st.sidebar.header("🔧 Connection")
api_in = st.sidebar.text_input("API Base URL", value=st.session_state.api_base, key="sb_api")
if api_in != st.session_state.api_base:
    st.session_state.api_base = api_in.rstrip("/")

ok, msg, info = check_api()
if ok:
    st.sidebar.success(f"{msg} (v{info.get('version','?')})")
else:
    st.sidebar.error(f"API error: {msg}")
    st.sidebar.code("uvicorn main:app --reload", language="bash")

st.sidebar.header("🗂️ Sessions")
if st.session_state.sessions:
    labels = [f"{s['filename']}  [{s['session_id'][:8]}]" for s in st.session_state.sessions]
    idx = st.sidebar.selectbox(
        "Current Session",
        options=list(range(len(labels))),
        format_func=lambda i: labels[i],
        key="sb_session_picker"
    )
    st.session_state.current_session_id = st.session_state.sessions[idx]["session_id"]
    st.sidebar.caption(f"Active session_id: `{st.session_state.current_session_id}`")
else:
    st.sidebar.info("No local sessions yet. Upload a PDF first.")

if st.sidebar.button("🔄 Refresh server sessions", use_container_width=True, key="sb_refresh"):
    refresh_remote_sessions_sidebar()
else:
    refresh_remote_sessions_sidebar()

# ============================== Tabs ==============================
tab_upload, tab_ask, tab_senti, tab_anom, tab_forecast, tab_strategy = st.tabs(
    ["📄 Upload", "❓ Ask", "💭 Sentiment", "🧭 Anomalies", "📈 Forecast", "🎯 Strategy"]
)

# -------- Upload --------
with tab_upload:
    st.subheader("Upload a financial PDF to create a session")
    up = st.file_uploader("Choose PDF", type=["pdf"], key="up_pdf")
    c1, c2 = st.columns([1, 1])
    with c1:
        auto_wait = st.toggle("Wait until processing completes", value=True, key="up_wait")
    with c2:
        st.write("")

    if up and st.button("🚀 Upload & Process", type="primary", key="up_go"):
        with st.spinner("Uploading..."):
            ok_u, umsg, session_id, task_id = upload_pdf(up)
        if not ok_u or not session_id:
            st.error(f"❌ {umsg}")
        else:
            st.success(f"✅ {umsg}")
            st.info(f"session_id: `{session_id}`  | task_id: `{task_id}`")
            st.session_state.sessions.insert(
                0,
                {"session_id": session_id, "filename": up.name, "when": time.strftime("%Y-%m-%d %H:%M:%S")}
            )
            st.session_state.current_session_id = session_id
            if auto_wait and task_id:
                ok2, msg2 = poll_status(task_id, max_wait_s=180)
                if ok2:
                    st.success("✅ Document is ready.")
                else:
                    st.warning(f"Processing not completed: {msg2}")

# -------- Ask --------
with tab_ask:
    st.subheader("Ask questions about the active PDF (session-scoped)")
    if not st.session_state.current_session_id:
        st.info("Upload a PDF first or select a session in the sidebar.")
    q = st.text_area(
        "Your question",
        height=100,
        placeholder="E.g., What was total revenue for the last quarter?",
        key="ask_q"
    )
    col_q1, col_q2 = st.columns([3, 1])
    with col_q2:
        top_k = st.number_input("Top K chunks", min_value=1, max_value=20, value=8, key="ask_topk")
    if st.button("🔍 Get Answer", type="primary", key="ask_go"):
        if not q.strip():
            st.warning("Ask something first 🙂")
        else:
            with st.spinner("Querying..."):
                r = do_query(q.strip(), st.session_state.current_session_id, top_k=top_k)
            if r.status_code == 200:
                data = r.json()
                st.success("Answer:")
                st.markdown(data.get("answer", ""))
                m1, m2, m3 = st.columns(3)
                m1.metric("Processing (ms)", data.get("processing_time_ms", 0))
                m2.metric("Chunks used", data.get("chunks_used", 0))
                m3.metric("Confidence", f"{data.get('confidence', 0) * 100:.1f}%")
            else:
                st.error(f"❌ {r.json().get('detail', f'HTTP {r.status_code}')}")

# -------- Sentiment --------
with tab_senti:
    st.subheader("Market Sentiment")
    mode = st.radio("Analyze sentiment for:", ["Use current session", "Custom text"], horizontal=True, key="senti_mode")
    c1, c2 = st.columns([2, 1])
    with c2:
        top_k_chunks = st.number_input("Chunks to sample (session)", min_value=3, max_value=50, value=12, key="senti_topk")

    text = ""
    sid = None
    if mode == "Custom text":
        text = st.text_area("Paste financial text", height=160, key="senti_text")
    else:
        sid = st.session_state.current_session_id
        if not sid:
            st.info("No active session. Upload a PDF or switch to Custom text.")

    if st.button("🎭 Analyze Sentiment", type="primary", key="senti_go"):
        with st.spinner("Analyzing..."):
            r = do_sentiment(text, sid, top_k_chunks=top_k_chunks)
        if r.status_code == 200:
            data = r.json()
            res = data.get("result", {})
            st.success(f"Label: **{res.get('label','N/A')}**")
            m1, m2, m3 = st.columns(3)
            m1.metric("Polarity", f"{res.get('polarity', 0.0):+.2f}")
            m2.metric("Confidence", f"{res.get('confidence', 0.0) * 100:.1f}%")
            m3.metric("Source", res.get("source", "-"))
            st.caption(f"Mode: {data.get('mode','-')} | Session: {data.get('session_id','-')} | Sampled chars: {data.get('sampled_chars',0)}")
            if res.get("rationale"):
                st.markdown("**Rationale**")
                st.write(res.get("rationale"))
        else:
            st.error(f"❌ {r.json().get('detail', f'HTTP {r.status_code}')}")

# -------- Anomalies --------
with tab_anom:
    st.subheader("Anomaly detection")
    mode_a = st.radio("Analyze:", ["Use current session", "Custom text"], horizontal=True, key="anom_mode")
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        top_k_chunks_anom = st.number_input("Chunks to sample (session)", min_value=5, max_value=80, value=20, key="anom_topk")
    with c2:
        yoy_pct = st.slider("YoY/Change threshold (%)", min_value=5, max_value=100, value=30, step=5, key="anom_yoy")
    with c3:
        margin_bps = st.slider("Margin threshold (bps)", min_value=50, max_value=2000, value=300, step=50, key="anom_bps")

    text_a = ""
    sid_a = None
    if mode_a == "Custom text":
        text_a = st.text_area("Paste section with comparable figures", height=160, key="anom_text")
    else:
        sid_a = st.session_state.current_session_id
        if not sid_a:
            st.info("No active session. Upload a PDF or switch to Custom text.")

    if st.button("🧭 Run Anomaly Detection", type="primary", key="anom_go"):
        with st.spinner("Scanning..."):
            r = do_anomaly(text_a, sid_a, top_k_chunks=top_k_chunks_anom,
                           yoy_threshold=yoy_pct / 100.0, margin_bps_threshold=margin_bps)
        if r.status_code == 200:
            data = r.json()
            st.success("Scan complete")
            st.markdown("**Summary**")
            st.write(data.get("summary", ""))

            items = data.get("items", [])
            if items:
                df = pd.DataFrame(items)
                order = ["is_anomaly", "metric", "unit", "period_1", "value_1", "period_2", "value_2", "pct_change", "direction", "reason"]
                df = df[[c for c in order if c in df.columns]]
                if "pct_change" in df.columns:
                    df["pct_change"] = df["pct_change"].apply(lambda x: f"{x*100:.1f}%" if isinstance(x, (float, int)) else x)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No items returned.")

            st.caption(f"Mode: {data.get('mode','-')} | Session: {data.get('session_id','-')} | Sampled chars: {data.get('sampled_chars',0)}")
            st.caption(f"Rules → YoY ≥ {data.get('rules',{}).get('yoy_threshold_pct',0)}%, Margin ≥ {data.get('rules',{}).get('margin_threshold_bps',0)} bps, Sign-flip flagged.")
        else:
            st.error(f"❌ {r.json().get('detail', f'HTTP {r.status_code}')}")

# -------- Forecast (Stage 2) --------
with tab_forecast:
    st.subheader("Stage 2 — Forecast & Trading Signals")
    c1, c2, c3, c4 = st.columns([1.2, 1.6, 1, 1])
    with c1:
        ticker = st.text_input("Ticker", value="AAPL", key="fc_ticker").strip().upper()
    with c2:
        today = date.today()
        default_start = today - timedelta(days=365)
        dr = st.date_input("Date range (start / end)", value=(default_start, today), key="fc_range")
        if isinstance(dr, (list, tuple)) and len(dr) == 2:
            start_date = dr[0].strftime("%Y-%m-%d")
            end_date = dr[1].strftime("%Y-%m-%d")
        else:
            start_date = end_date = None
    with c3:
        horizon = st.number_input("Horizon (days)", min_value=5, max_value=120, value=30, key="fc_horizon")
    with c4:
        outdir = st.text_input("Output dir", value="outputs", key="fc_outdir")

    if st.button("🏁 Run Forecast", type="primary", key="fc_go"):
        if not (start_date and end_date):
            st.warning("Pick a valid start/end date range.")
        else:
            with st.spinner("Running forecasting pipeline..."):
                r = do_forecast(ticker, start_date, end_date, horizon=horizon, outdir=outdir)
            if r.status_code == 200:
                data = r.json()
                st.success("Forecast complete")

                # Next action
                na = data.get("next_action", {})
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Action", na.get("action", "HOLD"))
                m2.metric("Target Price", f"${na.get('price', 0.0):,.2f}")
                m3.metric("Hold Days", na.get("holding_days", 0))
                m4.metric("Confidence", f"{na.get('confidence', 0.0) * 100:.1f}%")
                st.caption(na.get("reason", ""))

                # Links + embed chart
                chart_url = join_url(st.session_state.api_base, data.get("chart_url", ""))
                sig_url = join_url(st.session_state.api_base, data.get("signals_csv_url", ""))
                fc_url = join_url(st.session_state.api_base, data.get("forecasts_csv_url", ""))

                st.markdown(f"**Chart:** {chart_url}")
                st.markdown(f"**Signals CSV:** {sig_url}")
                st.markdown(f"**Forecasts CSV:** {fc_url}")

                with st.expander("📈 Preview Chart"):
                    st.components.v1.html(
                        f'<iframe src="{chart_url}" width="100%" height="950" frameborder="0"></iframe>',
                        height=970
                    )

                # Metrics
                st.markdown("### Model Metrics")
                metrics = data.get("metrics", {})
                if metrics:
                    dfm = pd.DataFrame.from_dict(metrics, orient="index", columns=["value"])
                    st.dataframe(dfm, use_container_width=True)
                # Notes
                notes = data.get("notes", [])
                if notes:
                    st.markdown("**Notes**")
                    for n in notes:
                        st.write("• " + n)
            else:
                st.error(f"❌ {r.json().get('detail', f'HTTP {r.status_code}')}")

# -------- Strategy (Stage 3) --------
with tab_strategy:
    st.subheader("Stage 3 — Strategy Recommendation")
    c1, c2, c3, c4 = st.columns([1.1, 1.6, 1, 1])
    with c1:
        st_ticker = st.text_input("Ticker", value="AAPL", key="st_ticker").strip().upper()
    with c2:
        today2 = date.today()
        default_start2 = today2 - timedelta(days=365)
        st_range = st.date_input("Date range (start / end)", value=(default_start2, today2), key="st_range")
        if isinstance(st_range, (list, tuple)) and len(st_range) == 2:
            st_start = st_range[0].strftime("%Y-%m-%d")
            st_end = st_range[1].strftime("%Y-%m-%d")
        else:
            st_start = st_end = None
    with c3:
        st_horizon = st.number_input("Horizon (days)", min_value=5, max_value=120, value=30, key="st_horizon")
    with c4:
        st_outdir = st.text_input("Output dir", value="outputs", key="st_outdir")

    c5, c6, c7 = st.columns([1, 1, 1])
    with c5:
        buy_thr = st.slider("BUY threshold (score)", min_value=0.10, max_value=0.80, value=0.30, step=0.05, key="st_buy_thr")
    with c6:
        sell_thr = st.slider("SELL threshold (score)", min_value=0.10, max_value=0.80, value=0.30, step=0.05, key="st_sell_thr")
    with c7:
        st_sid = st.text_input("Session ID (optional)", value=(st.session_state.current_session_id or ""), key="st_session_id")

    st.markdown("**Optional sentiment/anomaly override text** (if you don't want to use session chunks):")
    st_override = st.text_area("Paste text (optional)", height=120, key="st_override_text")

    if st.button("🚦 Get Strategy", type="primary", key="st_go"):
        if not (st_start and st_end):
            st.warning("Pick a valid start/end date range.")
        else:
            with st.spinner("Running forecast..."):
                fr = do_forecast(st_ticker, st_start, st_end, horizon=st_horizon, outdir=st_outdir)
            if fr.status_code != 200:
                st.error(f"Forecast error: {fr.json().get('detail', f'HTTP {fr.status_code}')}")  # stop early
            else:
                forecast_json = fr.json()

                # sentiment
                with st.spinner("Analyzing sentiment..."):
                    sr = do_sentiment(st_override if st_override.strip() else None,
                                      st_sid if not st_override.strip() else None,
                                      top_k_chunks=12)
                sentiment_json = sr.json() if sr.status_code == 200 else None

                # anomaly
                with st.spinner("Scanning anomalies..."):
                    ar = do_anomaly(st_override if st_override.strip() else None,
                                    st_sid if not st_override.strip() else None,
                                    top_k_chunks=20, yoy_threshold=0.30, margin_bps_threshold=300)
                anomalies_json = ar.json() if ar.status_code == 200 else None

                # Try backend strategy if available; else fallback
                payload = {
                    "ticker": st_ticker,
                    "start": st_start,
                    "end": st_end,
                    "horizon": int(st_horizon),
                    "outdir": st_outdir,
                    "session_id": st_sid or None,
                    "override_text": st_override.strip() or None,
                    "buy_threshold": float(buy_thr),
                    "sell_threshold": float(sell_thr)
                }
                ok_backend, strat = try_strategy_backend(payload)
                if ok_backend:
                    st.success("Strategy (backend)")
                    decision = strat.get("decision", "HOLD")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Decision", decision)
                    m2.metric("Score", f"{strat.get('score', 0.0):+.2f}")
                    m3.metric("Used Backend", "Yes")
                    for r in strat.get("reasons", []):
                        st.write("• " + r)
                else:
                    st.info("Backend strategy endpoint not available — using local strategy.")
                    res = local_strategy_fallback(
                        forecast_json=forecast_json,
                        sentiment_json=sentiment_json,
                        anomalies_json=anomalies_json,
                        buy_threshold=float(buy_thr),
                        sell_threshold=float(sell_thr),
                    )
                    decision = res["decision"]
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Decision", decision)
                    m2.metric("Score", f"{res['score']:+.2f}")
                    m3.metric("Used Backend", "No")
                    for r in res["reasons"]:
                        st.write("• " + r)

                # Show quick context
                st.markdown("---")
                st.markdown("**Context snapshots**")
                # Next action from forecast
                na = forecast_json.get("next_action", {}) if forecast_json else {}
                cna1, cna2, cna3, cna4 = st.columns(4)
                cna1.metric("Forecast→Action", na.get("action", "HOLD"))
                cna2.metric("Forecast→Price", f"${na.get('price', 0.0):,.2f}")
                cna3.metric("Forecast→Hold (d)", na.get("holding_days", 0))
                cna4.metric("Forecast→Conf", f"{na.get('confidence', 0.0) * 100:.1f}%")

                # Sentiment summary
                if sentiment_json:
                    res = sentiment_json.get("result", {})
                    cs1, cs2, cs3 = st.columns(3)
                    cs1.metric("Sentiment", res.get("label", "N/A"))
                    cs2.metric("Polarity", f"{res.get('polarity', 0.0):+.2f}")
                    cs3.metric("Confidence", f"{res.get('confidence', 0.0) * 100:.1f}%")

                # Anomaly count
                if anomalies_json:
                    items = anomalies_json.get("items", [])
                    cnt = sum(1 for it in items if it.get("is_anomaly"))
                    st.metric("Anomalies flagged", cnt)

                # Chart preview
                chart_url = join_url(st.session_state.api_base, forecast_json.get("chart_url", ""))
                with st.expander("📈 Preview Strategy Chart"):
                    st.components.v1.html(
                        f'<iframe src="{chart_url}" width="100%" height="950" frameborder="0"></iframe>',
                        height=970
                    )

# ============================== Footer ==============================
st.markdown("---")
st.caption("Frontend for AkashX.ai Hackathon • Talks to your FastAPI: /upload-pdf, /query, /sentiment, /anomaly, /forecast/run, [/strategy/recommend (optional)]")
