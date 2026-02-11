import time
import hashlib
import re
from statistics import mean
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple

from loguru import logger

from src.config import Config
from src.vector_store import VectorStore
from src.reranker import Reranker
from src.llm_client import LLMClient


class RAGPipeline:
    """
    RAG Pipeline with:
    - Vector retrieval
    - Optional reranking
    - LLM generation
    - In-memory cache (TTL + max size)
    - Metrics: avg latency E2E, avg latency uncached, cache hit rate
    - Confidence scoring + evidence (differentiator)
    """

    def __init__(self):
        logger.info("Init RAG Pipeline...")

        self.vector_store = VectorStore()
        if not self.vector_store.exists():
            logger.info("Creating vector database...")
            self.vector_store.create()

        self.reranker = Reranker()
        self.llm = LLMClient()

        # Cache: key -> (result_dict, timestamp)
        self.cache: Dict[str, Tuple[Dict[str, Any], datetime]] = {}

        # Metrics
        self.metrics = {
            "total_queries": 0,
            "cache_hits": 0,
            "total_latency_ms_e2e": 0,       # includes cached responses
            "total_latency_ms_uncached": 0,  # only real pipeline cost
            "uncached_queries": 0
        }

        logger.info("✓ Pipeline ready\n")

    # =========================
    # PUBLIC API
    # =========================
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        use_cache: bool = True,
        use_rerank: bool = True
    ) -> Dict[str, Any]:

        start = time.time()
        self.metrics["total_queries"] += 1

        question_clean = (question or "").strip()
        if not question_clean:
            return self._finalize(
                question="",
                answer="Empty question.",
                docs_scores=[],
                start=start,
                from_cache=False,
                allow_cache_write=False
            )

        # -------- Cache read --------
        if use_cache and Config.enable_cache:
            cached = self._get_cache(question_clean)
            if cached is not None:
                self.metrics["cache_hits"] += 1

                # measure E2E latency even for cache hits (network/UI will add more)
                latency_ms = int((time.time() - start) * 1000)
                self.metrics["total_latency_ms_e2e"] += latency_ms

                # return a fresh copy and overwrite fields that must be "now"
                out = dict(cached)
                out["from_cache"] = True
                out["latency_ms"] = latency_ms
                out["timestamp"] = datetime.now().isoformat()
                return out

        # -------- Retrieval --------
        k = int(top_k) if top_k is not None else int(Config.top_k_retrieval)
        docs_scores = self.vector_store.search(question_clean, k=k)

        if not docs_scores:
            return self._finalize(
                question=question_clean,
                answer="No relevant information found in the document.",
                docs_scores=[],
                start=start,
                from_cache=False,
                allow_cache_write=(use_cache and Config.enable_cache)
            )

        # -------- Reranking --------
        if use_rerank:
            docs_scores = self.reranker.rerank(
                question_clean, docs_scores, top_k=int(Config.top_k_final)
            )
        else:
            docs_scores = docs_scores[: int(Config.top_k_final)]

        # -------- Context --------
        context = self._build_context(docs_scores)

        # -------- LLM --------
        answer = self.llm.generate(context, question_clean)

        # -------- Finalize (+ cache write) --------
        return self._finalize(
            question=question_clean,
            answer=answer,
            docs_scores=docs_scores,
            start=start,
            from_cache=False,
            allow_cache_write=(use_cache and Config.enable_cache)
        )

    # =========================
    # FINAL RESPONSE BUILDER
    # =========================
    def _finalize(
        self,
        question: str,
        answer: str,
        docs_scores: List[Tuple[Any, float]],
        start: float,
        from_cache: bool,
        allow_cache_write: bool
    ) -> Dict[str, Any]:
        latency_ms = int((time.time() - start) * 1000)

        # E2E always counts
        self.metrics["total_latency_ms_e2e"] += latency_ms

        # Uncached only if not from cache
        if not from_cache:
            self.metrics["uncached_queries"] += 1
            self.metrics["total_latency_ms_uncached"] += latency_ms

        confidence = self._compute_confidence(docs_scores, answer)
        evidence = self._build_evidence(docs_scores)

        result = {
            "answer": answer,
            "sources": self._format_sources(docs_scores),
            "evidence": evidence,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "from_cache": from_cache,
            "timestamp": datetime.now().isoformat()
        }

        # Cache write must use QUESTION as key (NOT answer)
        if allow_cache_write and not from_cache:
            self._set_cache(question, result)

        logger.info(
            f"✓ Answer in {latency_ms} ms | cache={from_cache} | confidence={confidence.get('level')}"
        )
        return result

    # =========================
    # CONFIDENCE (DIFFERENTIATOR)
    # =========================
    def _compute_confidence(self, docs_scores: List[Tuple[Any, float]], answer: str) -> dict:
        if not docs_scores:
            return {"level": "LOW", "score": 0.0, "reasons": ["NO_SOURCES"]}

        scores = [float(s) for _, s in docs_scores if s is not None]
        if not scores:
            return {"level": "LOW", "score": 0.0, "reasons": ["NO_SCORES"]}

        top = max(scores)
        avg = mean(scores)

        has_numbers_answer = bool(re.search(r"\d", answer or ""))
        numeric_chunks = sum(1 for d, _ in docs_scores if (d.metadata or {}).get("has_numbers"))
        unique_pages = len({(d.metadata or {}).get("page") for d, _ in docs_scores})

        score = 0.0
        reasons = []

        # Similarity heuristics (adjust thresholds to your vector DB scale)
        if top >= 0.55:
            score += 0.40
        else:
            reasons.append("LOW_TOP_SIMILARITY")

        if avg >= 0.45:
            score += 0.20
        else:
            reasons.append("LOW_AVG_SIMILARITY")

        if unique_pages >= 2:
            score += 0.15
        else:
            reasons.append("SINGLE_PAGE_EVIDENCE")

        if numeric_chunks >= 2:
            score += 0.15
        else:
            reasons.append("LOW_NUMERIC_EVIDENCE")

        # Penalize “numbers with no numeric support”
        if has_numbers_answer and numeric_chunks == 0:
            score -= 0.20
            reasons.append("ANSWER_HAS_NUMBERS_WITHOUT_SUPPORT")

        score = max(0.0, min(1.0, score))
        level = "HIGH" if score >= 0.75 else "MEDIUM" if score >= 0.50 else "LOW"

        return {"level": level, "score": round(score, 3), "reasons": reasons}

    # =========================
    # EVIDENCE (AUDIT FRIENDLY)
    # =========================
    def _build_evidence(self, docs_scores: List[Tuple[Any, float]], max_items: int = 3) -> List[Dict[str, Any]]:
        evidence = []
        for doc, score in docs_scores[:max_items]:
            meta = doc.metadata or {}
            evidence.append({
                "page": meta.get("page", "?"),
                "score": round(float(score), 3),
                "snippet": (doc.page_content or "")[:240].strip()
            })
        return evidence

    # =========================
    # HELPERS
    # =========================
    def _build_context(self, docs_scores: List[Tuple[Any, float]]) -> str:
        parts = []
        for doc, score in docs_scores:
            meta = doc.metadata or {}
            page = meta.get("page", "?")
            parts.append(
                f"[Page {page} | score {float(score):.2f}]\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(parts)

    def _format_sources(self, docs_scores: List[Tuple[Any, float]]) -> List[Dict[str, Any]]:
        out = []
        for doc, score in docs_scores:
            meta = doc.metadata or {}
            out.append({
                "page": meta.get("page", "?"),
                "score": round(float(score), 3),
                "preview": ((doc.page_content or "")[:150] + "...")
            })
        return out

    # =========================
    # CACHE
    # =========================
    def _get_cache_key(self, question: str) -> str:
        return hashlib.md5(question.lower().strip().encode("utf-8")).hexdigest()

    def _get_cache(self, question: str) -> Optional[Dict[str, Any]]:
        key = self._get_cache_key(question)
        if key not in self.cache:
            return None

        cached, ts = self.cache[key]
        if datetime.now() - ts > timedelta(seconds=int(Config.cache_ttl)):
            del self.cache[key]
            return None

        return cached

    def _set_cache(self, question: str, result: Dict[str, Any]) -> None:
        key = self._get_cache_key(question)

        if len(self.cache) >= int(Config.cache_max_size):
            oldest_key = min(self.cache, key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]

        self.cache[key] = (result.copy(), datetime.now())

    # =========================
    # METRICS
    # =========================
    def get_metrics(self) -> Dict[str, Any]:
        tq = self.metrics["total_queries"]
        uq = self.metrics["uncached_queries"]

        avg_e2e = (self.metrics["total_latency_ms_e2e"] / tq) if tq else None
        avg_uncached = (self.metrics["total_latency_ms_uncached"] / uq) if uq else None
        cache_rate = (self.metrics["cache_hits"] / tq) if tq else 0.0

        return {
            "total_queries": tq,
            "cache_hits": self.metrics["cache_hits"],
            "cache_hit_rate": round(cache_rate, 3),
            "avg_latency_e2e_ms": round(avg_e2e, 2) if avg_e2e is not None else None,
            "avg_latency_uncached_ms": round(avg_uncached, 2) if avg_uncached is not None else None,
            "cache_size": len(self.cache),
            "db_stats": self.vector_store.stats()
        }
