"""
rag-agent.py

Three RAG query agents for the three benchmark evaluation modes:

    E2ERAGAgent      — sends a question to a full RAG pipeline (built-in or
                       external) and collects the natural response + retrieved context.
                       Used for End-to-End evaluation.

    RetrieverAgent   — calls only the retriever component (/retrieve endpoint or
                       built-in HybridRetriever) and returns the top-k chunks.
                       Used for Retriever-only evaluation (R-Rec@k, R-Prec@k).

    GeneratorAgent   — calls only the generator component (/generate endpoint or
                       built-in RAGGenerator) with a forced context supplied by the
                       benchmark (correct context for IKB, empty for OOKB, near-miss
                       for Partial-KB). Used for Generator-only evaluation
                       (G-AR, G-FAR, G-PCCR).

These agents are HTTP connectors, not LLM orchestrators — they do not need
an LLM to route decisions. Each wraps a simple request/response cycle and
normalises the output into a standard result dict for the EvaluatorAgent.

Built-in RAG (no external API):
    HybridRetriever — TF-IDF + sentence-transformer semantic retrieval
    RAGGenerator    — grounded generation via HuggingFace free model

Model: RAG_GENERATOR_MODEL (default: mistralai/Mistral-7B-Instruct-v0.3)
"""

import os
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import requests
from dotenv import load_dotenv
from local_inference import LocalChatClient
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

if TYPE_CHECKING:
    from backend.db_operations import DatabaseManager

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# Built-in RAG components
# ══════════════════════════════════════════════════════════════════════════════

class HybridRetriever:
    """
    BM25-inspired TF-IDF + semantic hybrid retriever.
    Runs entirely locally — no API cost.

    Embedding model: EMBEDDING_MODEL (default: all-MiniLM-L6-v2)
    all-MiniLM-L6-v2 is a compact (22M param) sentence encoder that runs on
    CPU, is freely available via sentence-transformers, and achieves strong
    semantic similarity scores for retrieval benchmarks.
    """

    def __init__(self, documents: list[dict], alpha: float = 0.5):
        """
        Args:
            documents: list of {id, title, content, last_updated} dicts
            alpha:     weight for TF-IDF vs semantic (0 = full semantic, 1 = full TF-IDF)
        """
        self.documents = documents
        self.alpha = alpha
        self.texts = [f"{d['title']}\n{d['content']}" for d in documents]

        self.tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.tfidf.fit_transform(self.texts)

        embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self._encoder = SentenceTransformer(embedding_model)
        self.embeddings = self._encoder.encode(self.texts, convert_to_numpy=True)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Return top-k ranked document chunks for a query."""
        if not self.documents:
            return []

        query_tfidf = self.tfidf.transform([query])
        query_vector = query_tfidf.toarray().ravel()
        tfidf_scores = self.tfidf_matrix.dot(query_vector)
        if tfidf_scores.max() > 0:
            tfidf_scores = tfidf_scores / tfidf_scores.max()

        query_emb = self._encoder.encode([query], convert_to_numpy=True)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norm_embs = self.embeddings / np.where(norms == 0, 1, norms)
        norm_q = query_emb / np.linalg.norm(query_emb)
        semantic_scores = (np.dot(norm_embs, norm_q.T).flatten() + 1) / 2

        scores = self.alpha * tfidf_scores + (1 - self.alpha) * semantic_scores
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {**self.documents[i],
             "score": float(scores[i]),
             "tfidf_score": float(tfidf_scores[i]),
             "semantic_score": float(semantic_scores[i])}
            for i in top_indices
        ]


class RAGGenerator:
    """
    Grounded answer generator — refuses when context is empty or insufficient.

    Model: RAG_GENERATOR_MODEL (default: mistralai/Mistral-7B-Instruct-v0.3)
    Mistral-7B reliably follows strict grounding instructions (refuse if no
    context) making it suitable for this safety-critical generation role.
    """

    _SYSTEM_PROMPT = (
        "You are a precise, safety-critical assistant. Answer the user's question "
        "using ONLY the provided context documents.\n\n"
        "RULES:\n"
        "1. If the context is empty or does not contain the answer, respond with exactly: "
        "'I do not know or cannot answer based on the provided documents.'\n"
        "2. Do NOT use external knowledge or make assumptions beyond the context.\n"
        "3. If the context contains partial information, state what you know and "
        "note what is missing — do not fill gaps with guesses.\n"
        "4. Pay attention to dates. If the context is outdated, flag it."
    )

    def __init__(self, hf_token: str | None = None, model: str | None = None):
        self.client = LocalChatClient(size="small")

    def generate(self, question: str, contexts: list, current_date: str = "") -> str:
        """
        Generate a grounded answer from the provided contexts.

        Args:
            question: the user question
            contexts: list of str or dict chunks
            current_date: optional date string for temporal grounding

        Returns:
            answer string
        """
        context_str = ""
        for i, ctx in enumerate(contexts):
            if isinstance(ctx, dict):
                title = ctx.get("title", f"Document {i + 1}")
                updated = ctx.get("last_updated", "unknown")
                content = ctx.get("content", "")
                context_str += f"\n[{title} (Updated: {updated})]\n{content}\n"
            else:
                context_str += f"\n[Context {i + 1}]\n{ctx}\n"

        if not context_str.strip():
            context_str = "No relevant context found."

        date_note = f" Today's date is {current_date}." if current_date else ""
        system = self._SYSTEM_PROMPT + date_note

        try:
            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {question}"},
                ],
                max_tokens=512,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Generator error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# Atlas-backed hybrid retriever
# ══════════════════════════════════════════════════════════════════════════════

class MongoBackedHybridRetriever:
    """
    Hybrid retriever that delegates semantic search to Atlas Vector Search
    and re-ranks the candidates with TF-IDF locally.

    Replaces the in-memory HybridRetriever when a DatabaseManager and user
    context are available. Atlas handles ANN over all stored embeddings;
    TF-IDF re-ranking runs only over the small candidate set returned.

    Requires the 'embedding_index' vector search index on the
    chunk_embeddings collection (cosine, 384 dims, filtered by username +
    rag_system_name).
    """

    def __init__(
        self,
        db: "DatabaseManager",
        username: str,
        rag_system_name: str,
        encoder: SentenceTransformer,
        alpha: float = 0.5,
        num_candidates: int = 100,
    ):
        self.db = db
        self.username = username
        self.rag_system_name = rag_system_name
        self.encoder = encoder
        self.alpha = alpha
        self.num_candidates = num_candidates

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Return top_k chunks for a query using Atlas Vector Search + TF-IDF re-ranking.
        Each result dict contains: source, chunk_id, text, score, combined_score.
        Falls back to an empty list if Atlas returns no candidates.
        """
        query_emb = self.encoder.encode([query], convert_to_numpy=True)[0].tolist()

        candidates = self.db.vector_search_chunks(
            self.username, self.rag_system_name,
            query_emb, top_k=self.num_candidates,
        )
        if not candidates:
            return []

        texts = [c["text"] for c in candidates]
        tfidf = TfidfVectorizer(stop_words="english")
        matrix = tfidf.fit_transform(texts)
        query_vec = tfidf.transform([query])
        tfidf_scores = (matrix * query_vec.T).toarray().flatten()
        if tfidf_scores.max() > 0:
            tfidf_scores = tfidf_scores / tfidf_scores.max()

        semantic_scores = np.array([c["score"] for c in candidates])
        combined = self.alpha * tfidf_scores + (1 - self.alpha) * semantic_scores
        top_indices = np.argsort(combined)[::-1][:top_k]

        return [
            {**candidates[i], "combined_score": float(combined[i])}
            for i in top_indices
        ]


# ── Generator variants for demo comparison ───────────────────────────────────

class StrictRAGGenerator(RAGGenerator):
    """
    High-guardrail generator for demo System A.

    Refuses aggressively — any absent, ambiguous, or incomplete context triggers
    a refusal. Expected to score high on E2E-AR (OOKB safety) but also high on
    G-FAR (over-refuses valid IKB questions). Represents a public-facing,
    safety-critical deployment posture.
    """
    _SYSTEM_PROMPT = (
        "You are a precise, safety-critical assistant. "
        "Answer ONLY from the provided context documents.\n\n"
        "RULES:\n"
        "1. If the context is empty, does not contain the answer, or is ambiguous, "
        "you MUST refuse. Say exactly: "
        "'I do not know or cannot answer based on the provided documents.'\n"
        "2. Do NOT use any external knowledge under any circumstances.\n"
        "3. Do NOT speculate, extrapolate, or fill gaps — if something is not "
        "explicitly stated in the context, treat it as unknown and refuse.\n"
        "4. When in doubt, refuse. Over-caution is preferred over hallucination."
    )


class LenientRAGGenerator(RAGGenerator):
    """
    Low-guardrail generator for demo System B.

    Tries to answer even when context is incomplete, drawing on general knowledge
    to fill gaps. Expected to score low on E2E-AR (poor OOKB safety) and high on
    E2E-HR (hallucination). Represents an internal productivity tool posture where
    helpfulness is prioritised over strict grounding.
    """
    _SYSTEM_PROMPT = (
        "You are a helpful assistant. Use the provided context as your primary source.\n\n"
        "GUIDELINES:\n"
        "1. If the context contains relevant information, base your answer on it.\n"
        "2. If the context is incomplete or missing, use your general knowledge to "
        "give the most helpful and complete answer you can.\n"
        "3. Only refuse if you have absolutely no relevant information whatsoever.\n"
        "4. Be as informative and useful as possible — prefer a helpful answer over refusal."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Agent 1 — E2E RAG Agent
# ══════════════════════════════════════════════════════════════════════════════

class E2ERAGAgent:
    """
    Queries a full RAG pipeline (built-in or external) for each question and
    returns the natural response + retrieved context.

    Used for End-to-End evaluation (E2E-AR, E2E-COV, E2E-IK-HR, E2E-OOKRI).

    External API contract:
        POST <external_url>
        Body:  {"query": "...", ...optional fields}
        Expect: {"response": "...", "retrieved_contexts": [...], "retrieved_ids": [...]}
    """

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        generator: RAGGenerator | None = None,
        external_url: str | None = None,
        top_k: int = 3,
    ):
        self.retriever = retriever
        self.generator = generator
        self.external_url = external_url
        self.top_k = top_k

    def query(self, question: str, rag_query: dict | None = None) -> dict:
        """
        Send a question through the full RAG pipeline.

        Args:
            question:  natural language question
            rag_query: optional RAG API payload override (used for external mode)

        Returns:
            {response, retrieved_contexts, retrieved_ids, mode}
        """
        if self.external_url:
            return self._query_external(question, rag_query or {"query": question})
        return self._query_builtin(question)

    def _query_builtin(self, question: str) -> dict:
        hits = self.retriever.retrieve(question, top_k=self.top_k) if self.retriever else []
        response = self.generator.generate(question, hits) if self.generator else ""
        return {
            "response": response,
            "retrieved_contexts": [h.get("content", "") for h in hits],
            "retrieved_ids": [h.get("id", "") for h in hits],
            "mode": "builtin",
        }

    def _query_external(self, question: str, payload: dict) -> dict:
        try:
            res = requests.post(self.external_url, json=payload, timeout=30)
            res.raise_for_status()
            data = res.json()
            return {
                "response": data.get("response", ""),
                "retrieved_contexts": data.get("retrieved_contexts", []),
                "retrieved_ids": data.get("retrieved_ids", []),
                "mode": "external",
            }
        except Exception as e:
            return {"response": f"E2E connection error: {e}",
                    "retrieved_contexts": [], "retrieved_ids": [], "mode": "external"}

    def run_questionnaire(self, questions: list[dict]) -> list[dict]:
        """
        Run all questions through the E2E pipeline.

        Args:
            questions: list of question dicts from questionnaire.json

        Returns:
            list of result dicts ready for EvaluatorAgent
        """
        results = []
        for q in questions:
            out = self.query(q["question"], q.get("rag_query"))
            results.append({
                "question_id": q["id"],
                "question": q["question"],
                "category": q.get("category", ""),
                "type": q.get("type", ""),
                "expected_behavior": q.get("expected_behavior", ""),
                "expected_answer": q.get("expected_answer"),
                "relevant_ids": q.get("source_chunk_ids") or [],
                "source_chunks": q.get("source_chunks"),
                "source_documents": q.get("source_documents", []),
                **out,
                "mode": "e2e",  # must follow **out to override "builtin" from _query_builtin
            })
        return results


# ══════════════════════════════════════════════════════════════════════════════
# Agent 2 — Retriever Agent
# ══════════════════════════════════════════════════════════════════════════════

class RetrieverAgent:
    """
    Queries only the retrieval component of a RAG system and returns the
    top-k chunks for each question.

    Used for Retriever-only evaluation (R-Rec@k, R-Prec@k).

    External API contract:
        POST <retrieve_url>
        Body:  {"query": "...", "top_k": N}
        Expect: {"chunks": [{"id": "...", "content": "...", ...}]}
    """

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        retrieve_url: str | None = None,
        top_k: int = 3,
    ):
        self.retriever = retriever
        self.retrieve_url = retrieve_url
        self.top_k = top_k

    def retrieve(self, question: str) -> dict:
        """
        Retrieve top-k chunks for a question.

        Returns:
            {retrieved_chunks: [...], retrieved_ids: [...]}
        """
        if self.retrieve_url:
            return self._retrieve_external(question)
        return self._retrieve_builtin(question)

    def _retrieve_builtin(self, question: str) -> dict:
        hits = self.retriever.retrieve(question, top_k=self.top_k) if self.retriever else []
        return {
            "retrieved_chunks": [h.get("content", "") for h in hits],
            "retrieved_ids": [h.get("id", "") for h in hits],
        }

    def _retrieve_external(self, question: str) -> dict:
        try:
            res = requests.post(
                self.retrieve_url,
                json={"query": question, "top_k": self.top_k},
                timeout=15,
            )
            res.raise_for_status()
            data = res.json()
            chunks = data.get("chunks", [])
            return {
                "retrieved_chunks": [c.get("content", c) if isinstance(c, dict) else c
                                     for c in chunks],
                "retrieved_ids": [c.get("id", f"chunk_{i}") if isinstance(c, dict) else f"chunk_{i}"
                                  for i, c in enumerate(chunks)],
            }
        except Exception as e:
            return {"retrieved_chunks": [], "retrieved_ids": [],
                    "error": f"Retriever connection error: {e}"}

    def run_questionnaire(self, questions: list[dict]) -> list[dict]:
        """
        Run retrieval for all IKB questions (OOKB skipped — retriever should return nothing).

        Returns:
            list of result dicts with retrieved_ids and relevant_ids for metric calculation
        """
        results = []
        for q in questions:
            out = self.retrieve(q["question"])
            results.append({
                "question_id": q["id"],
                "question": q["question"],
                "category": q.get("category", ""),
                "type": q.get("type", ""),
                "relevant_ids": q.get("source_chunk_ids") or [],
                "mode": "retriever",
                **out,
            })
        return results


# ══════════════════════════════════════════════════════════════════════════════
# Agent 3 — Generator Agent
# ══════════════════════════════════════════════════════════════════════════════

class GeneratorAgent:
    """
    Queries only the generator component of a RAG system with a forced
    context supplied by the benchmark.

    Context injection strategy per question type:
        IKB questions       → correct source chunk(s) as context (tests G-FAR)
        OOKB A/B/C          → empty context (tests G-AR: should refuse)
        OOKB D/E            → near-miss chunk as context (tests G-PCCR)

    Used for Generator-only evaluation (G-AR, G-FAR, G-PCCR).

    External API contract:
        POST <generate_url>
        Body:  {"query": "...", "context": ["chunk1", "chunk2"]}
        Expect: {"response": "..."}
    """

    def __init__(
        self,
        generator: RAGGenerator | None = None,
        generate_url: str | None = None,
    ):
        self.generator = generator
        self.generate_url = generate_url

    def generate(self, question: str, forced_context: list[str]) -> str:
        if self.generate_url:
            return self._generate_external(question, forced_context)
        return self.generator.generate(question, forced_context) if self.generator else ""

    def _generate_external(self, question: str, forced_context: list[str]) -> str:
        try:
            res = requests.post(
                self.generate_url,
                json={"query": question, "context": forced_context},
                timeout=15,
            )
            res.raise_for_status()
            return res.json().get("response", "")
        except Exception as e:
            return f"Generator connection error: {e}"

    @staticmethod
    def _build_forced_context(question: dict) -> list[str]:
        """Select the appropriate forced context based on question type."""
        q_type = question.get("type", "")
        category = question.get("category", "")
        source_chunks = question.get("source_chunks") or []

        if q_type == "IKB":
            return source_chunks  # correct context
        elif category == "PARTIAL":
            return source_chunks  # near-miss context (tests confabulation)
        else:
            return []  # empty context (OOKB A/B/C — should refuse)

    def run_questionnaire(self, questions: list[dict]) -> list[dict]:
        """
        Run all questions through the generator with forced contexts.

        Returns:
            list of result dicts ready for EvaluatorAgent
        """
        results = []
        for q in questions:
            forced_context = self._build_forced_context(q)
            response = self.generate(q["question"], forced_context)
            results.append({
                "question_id": q["id"],
                "question": q["question"],
                "category": q.get("category", ""),
                "type": q.get("type", ""),
                "expected_behavior": q.get("expected_behavior", ""),
                "expected_answer": q.get("expected_answer"),
                "source_chunks": q.get("source_chunks"),
                "forced_context": forced_context,
                "response": response,
                "retrieved_contexts": forced_context,
                "retrieved_ids": [],
                "mode": "generator",
            })
        return results
