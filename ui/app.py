import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, List

import requests
import streamlit as st


# =========================
# Page setup
# =========================
st.set_page_config(
    page_title="LVMH Financial Intelligence",
    page_icon="🚀",
    layout="wide",
)

# =========================
# CSS (clean + finance)
# =========================
st.markdown("""
<style>
.small-muted { color: #6B7280; font-size: 0.95rem; }
.section-title { font-size: 1.25rem; font-weight: 800; margin-top: 0.5rem; }
.card {
    background-color: #FFFFFF;
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #E5E7EB;
    box-shadow: 0 8px 30px rgba(0,0,0,0.06);
}
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 40%, #0b3a4a 100%);
    border-radius: 22px;
    padding: 28px;
    color: #E5E7EB;
    box-shadow: 0 30px 80px rgba(0,0,0,0.22);
}
.hero-title { font-size: 3.1rem; font-weight: 900; letter-spacing: -0.03em; margin: 0; }
.hero-sub { margin-top: 10px; color: #cbd5e1; font-size: 1.05rem; line-height: 1.6; }
.pill {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    border-radius: 999px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.10);
    margin-right: 10px;
    margin-top: 14px;
    font-weight: 700;
}
.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.dot-green { background: #10b981; box-shadow: 0 0 12px #10b981; }
.dot-red { background: #ef4444; box-shadow: 0 0 12px #ef4444; }

.kpi-wrap { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }
.kpi {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 18px;
    padding: 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.06);
}
.kpi-label { color: #64748b; font-size: 0.78rem; font-weight: 800; letter-spacing: 0.12em; text-transform: uppercase; }
.kpi-value { font-size: 2.2rem; font-weight: 900; color: #0f172a; margin-top: 8px; }
.kpi-sub { color: #64748b; font-size: 0.92rem; margin-top: 6px; }

.answer {
    background: #FFFFFF;
    border-radius: 18px;
    padding: 18px 18px;
    border: 1px solid #E5E7EB;
    box-shadow: 0 12px 35px rgba(0,0,0,0.07);
    border-left: 6px solid #2563eb;
}

.badge {
    display:inline-flex; align-items:center; gap:8px;
    padding: 8px 12px; border-radius: 999px;
    border: 1px solid #E5E7EB;
    background: #F8FAFC;
    font-weight: 800;
}
.badge-high { border-color: rgba(16,185,129,0.4); background: rgba(16,185,129,0.12); color:#065f46; }
.badge-med { border-color: rgba(245,158,11,0.4); background: rgba(245,158,11,0.14); color:#92400e; }
.badge-low { border-color: rgba(239,68,68,0.4); background: rgba(239,68,68,0.12); color:#7f1d1d; }

hr { border: none; border-top: 1px solid #E5E7EB; margin: 1.1rem 0; }
</style>
""", unsafe_allow_html=True)

# =========================
# API config
# =========================
DEFAULT_API = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")


def api_is_up(api_base: str) -> bool:
    try:
        r = requests.get(f"{api_base}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def call_query(api_base: str, payload: dict, timeout_s: int) -> dict:
    t0 = time.time()
    r = requests.post(f"{api_base}/query", json=payload, timeout=timeout_s)
    elapsed_ms = int((time.time() - t0) * 1000)
    if r.status_code != 200:
        raise RuntimeError(f"API error {r.status_code}: {r.text}")
    data = r.json()
    data.setdefault("latency_ms", elapsed_ms)
    return data


def call_metrics(api_base: str) -> dict:
    r = requests.get(f"{api_base}/metrics", timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"Metrics error {r.status_code}: {r.text}")
    return r.json()


def fmt_ms(x: Optional[float]) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):,.0f} ms"
    except Exception:
        return "—"


def confidence_badge(conf: Dict[str, Any]) -> str:
    level = (conf or {}).get("level", "LOW")
    score = (conf or {}).get("score", 0.0)

    if level == "HIGH":
        klass = "badge badge-high"
    elif level == "MEDIUM":
        klass = "badge badge-med"
    else:
        klass = "badge badge-low"

    return f"<span class='{klass}'>CONFIDENCE: {level} &nbsp;•&nbsp; Score: {score:.3f}</span>"


# =========================
# Session state
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.title("⚙️ Settings")
    api_base = st.text_input("API Base URL", value=DEFAULT_API)
    timeout_s = st.slider("Timeout (seconds)", 5, 60, 30)

    st.divider()
    st.subheader("Query options")
    top_k = st.number_input("Top-K", min_value=1, max_value=20, value=5, step=1)
    use_rerank = st.toggle("Use reranking", value=True)
    use_cache = st.toggle("Use cache", value=True)
    show_sources = st.toggle("Show sources", value=True)
    show_evidence = st.toggle("Show evidence (top 3)", value=True)

    st.divider()
    view_mode = st.radio("Page", options=["Executive Summary", "Ask (Chat)"], index=0)

# =========================
# Header / Hero
# =========================
up = api_is_up(api_base)

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-title">🚀 LVMH Financial Intelligence</div>
        <div class="hero-sub">
            Executive-grade RAG dashboard for corporate reporting — multi-language (FR/EN),
            traceable answers with sources, and performance analytics (latency, cache).
        </div>
        <div>
            <span class="pill">
                <span class="dot {'dot-green' if up else 'dot-red'}"></span>
                {'API ONLINE' if up else 'API OFFLINE'}
            </span>
            <span class="pill">RAG • Vector Search → Rerank → LLM</span>
            <span class="pill">Finance-ready: sources + audit trail</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# =========================
# Executive Summary
# =========================
if view_mode == "Executive Summary":
    st.markdown("<div class='section-title'>Executive Summary</div>", unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>High-level KPIs + system performance. Designed for stakeholder demos.</div>", unsafe_allow_html=True)
    st.write("")

    if not up:
        st.warning("API is offline. Start it first: `uvicorn api.app:app --host 0.0.0.0 --port 8000`")
    else:
        try:
            m = call_metrics(api_base)
        except Exception as e:
            st.error(f"Could not load /metrics: {e}")
            m = {}

        db_stats = (m.get("db_stats") or {})
        total_docs = db_stats.get("total_docs", 0)

        avg_e2e = m.get("avg_latency_e2e_ms", None)
        avg_uncached = m.get("avg_latency_uncached_ms", None)
        cache_rate = m.get("cache_hit_rate", 0.0)
        cache_size = m.get("cache_size", 0)

        st.markdown(
            f"""
            <div class="kpi-wrap">
                <div class="kpi">
                    <div class="kpi-label">Vector DB</div>
                    <div class="kpi-value">{int(total_docs):,}</div>
                    <div class="kpi-sub">chunks indexed</div>
                </div>
                <div class="kpi">
                    <div class="kpi-label">Avg Latency (E2E)</div>
                    <div class="kpi-value">{fmt_ms(avg_e2e)}</div>
                    <div class="kpi-sub">includes cache hits</div>
                </div>
                <div class="kpi">
                    <div class="kpi-label">Avg Latency (Uncached)</div>
                    <div class="kpi-value">{fmt_ms(avg_uncached)}</div>
                    <div class="kpi-sub">real pipeline cost (no cache)</div>
                </div>
                <div class="kpi">
                    <div class="kpi-label">Cache Hit Rate</div>
                    <div class="kpi-value">{cache_rate*100:.1f}%</div>
                    <div class="kpi-sub">cache size: {cache_size}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"**Embedding model:** `{db_stats.get('model', 'unknown')}`")
        st.markdown("**Pipeline:** Vector Search → Reranking → LLM Generation")
        st.markdown("**Traceability:** sources + evidence + confidence scoring")
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.markdown("<div class='section-title'>Quick Demo</div>", unsafe_allow_html=True)
        st.markdown("<div class='small-muted'>One-click questions to show value quickly.</div>", unsafe_allow_html=True)

        demo_qs = [
            "What was LVMH revenue in 2023?",
            "How many stores does LVMH have worldwide?",
            "Quels sont les principaux marchés géographiques ?",
        ]
        cols = st.columns(3)
        clicked = None
        for i in range(3):
            if cols[i].button(demo_qs[i], use_container_width=True):
                clicked = demo_qs[i]

        if clicked:
            payload = {
                "question": clicked,
                "top_k": int(top_k),
                "use_rerank": bool(use_rerank),
                "use_cache": bool(use_cache),
            }
            try:
                data = call_query(api_base, payload, timeout_s=timeout_s)
                answer = data.get("answer", "")
                conf = data.get("confidence", {}) or {}

                st.write("")
                st.markdown(confidence_badge(conf), unsafe_allow_html=True)
                st.markdown(f"<div class='answer'>{answer}</div>", unsafe_allow_html=True)

                st.caption(
                    f"Latency: {data.get('latency_ms','?')} ms | "
                    f"Cache: {'HIT' if data.get('from_cache') else 'MISS'} | "
                    f"Sources: {len(data.get('sources', []))}"
                )

                st.session_state.history.insert(0, {
                    "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "q": clicked,
                    "answer": answer,
                    "meta": {
                        "latency_ms": data.get("latency_ms", 0),
                        "from_cache": data.get("from_cache", False),
                        "sources": data.get("sources", []),
                        "evidence": data.get("evidence", []),
                        "confidence": conf,
                    }
                })

                if show_evidence:
                    ev = data.get("evidence", []) or []
                    for i, e in enumerate(ev[:3], 1):
                        with st.expander(f"Evidence {i} — Page {e.get('page','?')} — score {e.get('score','?')}"):
                            st.write(e.get("snippet", ""))

                if show_sources:
                    sources = data.get("sources", []) or []
                    for i, s in enumerate(sources[:8], 1):
                        with st.expander(f"Source {i} — Page {s.get('page','?')} — score {s.get('score','?')}"):
                            st.write(s.get("preview", ""))

            except Exception as e:
                st.error(f"Query failed: {e}")

        st.write("")
        st.markdown("<div class='section-title'>Timeline</div>", unsafe_allow_html=True)
        if not st.session_state.history:
            st.caption("No questions yet.")
        else:
            for item in st.session_state.history[:10]:
                with st.expander(f"{item['ts']} — {item['q']}"):
                    st.write(item["answer"])
                    meta = item["meta"]
                    st.caption(
                        f"Latency: {meta.get('latency_ms','?')} ms | "
                        f"Cache: {'HIT' if meta.get('from_cache') else 'MISS'} | "
                        f"Sources: {len(meta.get('sources', []))}"
                    )

# =========================
# Chat
# =========================
else:
    st.markdown("<div class='section-title'>Ask (Chat)</div>", unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>Ask questions in French or English. Answers come with sources.</div>", unsafe_allow_html=True)
    st.write("")

    if not up:
        st.warning("API is offline. Start it: `uvicorn api.app:app --host 0.0.0.0 --port 8000`")
    else:
        question = st.chat_input("Type your question (FR/EN)...")
        if question:
            payload = {
                "question": question,
                "top_k": int(top_k),
                "use_rerank": bool(use_rerank),
                "use_cache": bool(use_cache),
            }
            try:
                data = call_query(api_base, payload, timeout_s=timeout_s)
                answer = data.get("answer", "")
                conf = data.get("confidence", {}) or {}

                st.markdown(confidence_badge(conf), unsafe_allow_html=True)
                st.markdown(
                    f"<div class='answer'><b>Q:</b> {question}<br/><b>A:</b> {answer}</div>",
                    unsafe_allow_html=True
                )

                c1, c2, c3 = st.columns(3)
                c1.metric("Latency (ms)", int(data.get("latency_ms", 0)))
                c2.metric("Cache", "HIT" if data.get("from_cache") else "MISS")
                c3.metric("Sources", len(data.get("sources", []) or []))

                if show_evidence:
                    ev = data.get("evidence", []) or []
                    for i, e in enumerate(ev[:3], 1):
                        with st.expander(f"Evidence {i} — Page {e.get('page','?')} — score {e.get('score','?')}"):
                            st.write(e.get("snippet", ""))

                if show_sources:
                    sources = data.get("sources", []) or []
                    for i, s in enumerate(sources[:8], 1):
                        with st.expander(f"Source {i} — Page {s.get('page','?')} — score {s.get('score','?')}"):
                            st.write(s.get("preview", ""))

                st.session_state.history.insert(0, {
                    "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "q": question,
                    "answer": answer,
                    "meta": {
                        "latency_ms": data.get("latency_ms", 0),
                        "from_cache": data.get("from_cache", False),
                        "sources": data.get("sources", []),
                        "evidence": data.get("evidence", []),
                        "confidence": conf,
                    }
                })

            except Exception as e:
                st.error(f"Query failed: {e}")

    st.write("")
    a, b = st.columns(2)
    if a.button("🧹 Clear timeline", use_container_width=True):
        st.session_state.history = []
        st.success("Timeline cleared.")
    if b.button("📖 Open API docs", use_container_width=True):
        st.markdown(f"{api_base}/docs")
