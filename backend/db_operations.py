"""
db_operations.py

DatabaseManager — centralised MongoDB persistence layer for the OOKB benchmark.

Collections managed:
    users              — authentication (username / password_hash)
    chunk_embeddings   — KB text chunks + sentence-transformer embeddings
    questionnaire      — IKB/OOKB evaluation questions
    evaluation_results — per-question judge outputs across benchmark runs
    metrics            — aggregated benchmark metrics per run

Every collection except users stores username and rag_system_name so multiple
users can benchmark multiple RAG systems within the same database.

Requires:
    pip install pymongo sentence-transformers werkzeug python-dotenv
    MONGODB_URI, MONGODB_DB in .env
"""

import os
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, MongoClient, UpdateOne
from pymongo.collection import Collection
from werkzeug.security import check_password_hash, generate_password_hash

load_dotenv()

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_run_id() -> str:
    return f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


class DatabaseManager:
    """
    Central persistence layer — wraps all OOKB benchmark collections plus a
    users collection for authentication.

    Usage:
        db = DatabaseManager()

        # Auth
        db.register_user("alice", "s3cr3t")
        ok = db.verify_user("alice", "s3cr3t")   # True

        # Chunk embeddings
        db.upsert_chunk_embeddings("alice", "my_rag", chunks, encoder)

        # Questionnaire
        db.upsert_questionnaire("alice", "my_rag", questions)

        # Evaluation results
        run_id = db.insert_evaluation_results("alice", "my_rag", results)

        # Metrics
        db.upsert_metrics("alice", "my_rag", run_id, metrics, insights)
    """

    def __init__(self, uri: str | None = None, db_name: str | None = None):
        self._uri = uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        self._db_name = db_name or os.getenv("MONGODB_DB", "ookb_benchmark")

        self._client = MongoClient(
            self._uri,
            maxIdleTimeMS=45_000,
            connectTimeoutMS=10_000,
            serverSelectionTimeoutMS=10_000,
            retryReads=True,
            retryWrites=True,
        )
        _db = self._client[self._db_name]

        self._users: Collection = _db["users"]
        self._chunks: Collection = _db["chunk_embeddings"]
        self._questionnaire: Collection = _db["questionnaire"]
        self._eval_results: Collection = _db["evaluation_results"]
        self._metrics: Collection = _db["metrics"]

        self._create_indexes()

    # ── Index setup ────────────────────────────────────────────────────────────

    def _create_indexes(self) -> None:
        self._users.create_index("username", unique=True, background=True)

        self._chunks.create_index(
            [("username", ASCENDING), ("rag_system_name", ASCENDING),
             ("source", ASCENDING), ("chunk_id", ASCENDING)],
            unique=True, background=True,
        )

        self._questionnaire.create_index(
            [("username", ASCENDING), ("rag_system_name", ASCENDING), ("id", ASCENDING)],
            unique=True, background=True,
        )

        # Allow querying all results for a run
        self._eval_results.create_index(
            [("username", ASCENDING), ("evaluation_run_name", ASCENDING), ("run_id", ASCENDING)],
            background=True,
        )
        # Unique per question + mode within a run — makes re-runs idempotent
        self._eval_results.create_index(
            [("username", ASCENDING), ("evaluation_run_name", ASCENDING),
             ("run_id", ASCENDING), ("question_id", ASCENDING), ("mode", ASCENDING)],
            unique=True, background=True,
        )
        self._eval_results.create_index(
            [("username", ASCENDING), ("evaluation_run_name", ASCENDING), ("created_at", DESCENDING)],
            background=True,
        )

        self._metrics.create_index(
            [("username", ASCENDING), ("evaluation_run_name", ASCENDING), ("run_id", ASCENDING)],
            unique=True, background=True,
        )
        self._metrics.create_index(
            [("username", ASCENDING), ("evaluation_run_name", ASCENDING), ("created_at", DESCENDING)],
            background=True,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Users / Authentication
    # ══════════════════════════════════════════════════════════════════════════

    def register_user(self, username: str, password: str) -> bool:
        """
        Register a new user. Returns True on success, False if username already exists.
        Passwords are stored as bcrypt hashes via werkzeug.
        """
        try:
            self._users.insert_one({
                "username": username,
                "password_hash": generate_password_hash(password),
                "created_at": _now(),
            })
            return True
        except Exception:
            # DuplicateKeyError when username is taken
            return False

    def verify_user(self, username: str, password: str) -> bool:
        """Return True if username exists and password matches the stored hash."""
        doc = self._users.find_one({"username": username}, {"password_hash": 1})
        if not doc:
            return False
        return check_password_hash(doc["password_hash"], password)

    def user_exists(self, username: str) -> bool:
        return self._users.count_documents({"username": username}) > 0

    # ══════════════════════════════════════════════════════════════════════════
    # Chunk Embeddings
    # ══════════════════════════════════════════════════════════════════════════

    def upsert_chunk_embeddings(
        self,
        username: str,
        rag_system_name: str,
        chunks: list[dict],
        encoder: "SentenceTransformer",
        batch_size: int = 64,
    ) -> int:
        """
        Encode chunks and upsert into the chunk_embeddings collection.

        Args:
            username:        owning user
            rag_system_name: name of the RAG system being benchmarked
            chunks:          list of {source, chunk_id, text} dicts from DocumentProcessingAgent
            encoder:         SentenceTransformer instance (e.g. all-MiniLM-L6-v2)
            batch_size:      encoding batch size (tune for available VRAM/RAM)

        Returns:
            total documents upserted + modified
        """
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        print(f"  Encoding {len(texts)} chunks for [{username}/{rag_system_name}]...")
        embeddings = encoder.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        now = _now()
        ops = []
        for chunk, emb in zip(chunks, embeddings):
            ops.append(UpdateOne(
                filter={
                    "username": username,
                    "rag_system_name": rag_system_name,
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"],
                },
                update={
                    "$set": {
                        "username": username,
                        "rag_system_name": rag_system_name,
                        "source": chunk["source"],
                        "chunk_id": chunk["chunk_id"],
                        "text": chunk["text"],
                        "embedding": emb.tolist(),
                        "updated_at": now,
                    },
                    "$setOnInsert": {"created_at": now},
                },
                upsert=True,
            ))

        result = self._chunks.bulk_write(ops, ordered=False)
        total = result.upserted_count + result.modified_count
        print(f"  Chunk embeddings: {result.upserted_count} new, {result.modified_count} updated")
        return total

    def load_chunk_embeddings(self, username: str, rag_system_name: str) -> list[dict]:
        """Return all chunks for a user/system as {source, chunk_id, text, embedding}."""
        return [
            {
                "source":    doc["source"],
                "chunk_id":  doc["chunk_id"],
                "text":      doc["text"],
                "embedding": doc["embedding"],
            }
            for doc in self._chunks.find(
                {"username": username, "rag_system_name": rag_system_name},
                {"_id": 0, "source": 1, "chunk_id": 1, "text": 1, "embedding": 1},
            )
        ]

    def count_chunk_embeddings(self, username: str, rag_system_name: str) -> int:
        """Return the number of stored chunks for a user/system (cheap count query)."""
        return self._chunks.count_documents(
            {"username": username, "rag_system_name": rag_system_name}
        )

    def delete_chunk_embeddings(self, username: str, rag_system_name: str) -> int:
        """Delete all chunk embeddings for a user/system. Returns deleted count."""
        result = self._chunks.delete_many(
            {"username": username, "rag_system_name": rag_system_name}
        )
        print(f"  Deleted {result.deleted_count} chunk embeddings for [{username}/{rag_system_name}]")
        return result.deleted_count

    # ══════════════════════════════════════════════════════════════════════════
    # Questionnaire
    # ══════════════════════════════════════════════════════════════════════════

    def upsert_questionnaire(
        self,
        username: str,
        rag_system_name: str,
        questions: list[dict],
    ) -> int:
        """
        Upsert evaluation questions into the questionnaire collection.
        Each question is keyed by (username, rag_system_name, id).

        Args:
            username:        owning user
            rag_system_name: name of the RAG system
            questions:       list of question dicts from QuestionGenerationAgent
                             — each must have an "id" field (e.g. "q_001")

        Returns:
            total documents upserted + modified
        """
        if not questions:
            return 0

        now = _now()
        ops = []
        for q in questions:
            ops.append(UpdateOne(
                filter={
                    "username": username,
                    "rag_system_name": rag_system_name,
                    "id": q["id"],
                },
                update={
                    "$set": {
                        "username": username,
                        "rag_system_name": rag_system_name,
                        "id": q["id"],
                        "question": q.get("question", ""),
                        "type": q.get("type", ""),
                        "category": q.get("category", ""),
                        "difficulty": q.get("difficulty", ""),
                        "expected_behavior": q.get("expected_behavior", ""),
                        "expected_answer": q.get("expected_answer"),
                        "source_documents": q.get("source_documents", []),
                        "source_chunks": q.get("source_chunks"),
                        "rag_query": q.get("rag_query", {}),
                        "updated_at": now,
                    },
                    "$setOnInsert": {"created_at": now},
                },
                upsert=True,
            ))

        result = self._questionnaire.bulk_write(ops, ordered=False)
        total = result.upserted_count + result.modified_count
        print(f"  Questionnaire: {result.upserted_count} new, {result.modified_count} updated "
              f"for [{username}/{rag_system_name}]")
        return total

    def load_questionnaire(self, username: str, rag_system_name: str) -> list[dict]:
        """Return all questions for a user/system, sorted by id."""
        return list(self._questionnaire.find(
            {"username": username, "rag_system_name": rag_system_name},
            {"_id": 0},
            sort=[("id", ASCENDING)],
        ))

    def delete_questionnaire(self, username: str, rag_system_name: str) -> int:
        """Delete all questions for a user/system. Returns deleted count."""
        result = self._questionnaire.delete_many(
            {"username": username, "rag_system_name": rag_system_name}
        )
        return result.deleted_count

    # ══════════════════════════════════════════════════════════════════════════
    # Evaluation Results
    # ══════════════════════════════════════════════════════════════════════════

    def insert_evaluation_results(
        self,
        username: str,
        evaluation_run_name: str,
        results: list[dict],
        run_id: str | None = None,
    ) -> str:
        """
        Insert per-question evaluation results for a benchmark run.
        Uses upsert on (username, evaluation_run_name, run_id, question_id, mode)
        so re-submitting the same run is idempotent.

        Args:
            username:             owning user
            evaluation_run_name:  e.g. "my-rag-system_strict"
            results:              list of result dicts from JudgeAgent.evaluate_results()
                                  — each must have question_id and mode fields
            run_id:               optional run identifier; auto-generated if None

        Returns:
            run_id string (use this when calling upsert_metrics)
        """
        run_id = run_id or _new_run_id()
        if not results:
            return run_id

        now = _now()
        ops = []
        for r in results:
            ops.append(UpdateOne(
                filter={
                    "username": username,
                    "evaluation_run_name": evaluation_run_name,
                    "run_id": run_id,
                    "question_id": r.get("question_id", ""),
                    "mode": r.get("mode", ""),
                },
                update={"$set": {
                    "username": username,
                    "evaluation_run_name": evaluation_run_name,
                    "run_id": run_id,
                    "question_id": r.get("question_id", ""),
                    "question": r.get("question", ""),
                    "category": r.get("category", ""),
                    "type": r.get("type", ""),
                    "mode": r.get("mode", ""),
                    "expected_behavior": r.get("expected_behavior", ""),
                    "expected_answer": r.get("expected_answer"),
                    "response": r.get("response", ""),
                    "retrieved_ids": r.get("retrieved_ids", []),
                    "retrieved_contexts": r.get("retrieved_contexts", []),
                    "relevant_ids": r.get("relevant_ids", []),
                    "refusal_detected": r.get("refusal_detected", False),
                    "factual_alignment": r.get("factual_alignment", ""),
                    "context_grounding": r.get("context_grounding", ""),
                    "reasoning": r.get("reasoning", ""),
                    "final_classification": r.get("final_classification", ""),
                    "created_at": now,
                }},
                upsert=True,
            ))

        self._eval_results.bulk_write(ops, ordered=False)
        print(f"  Evaluation results: {len(results)} results stored for run [{run_id}]")
        return run_id

    def load_evaluation_results(
        self,
        username: str,
        evaluation_run_name: str,
        run_id: str | None = None,
    ) -> list[dict]:
        """
        Load evaluation results for a run.
        If run_id is None, returns results from the most recent run.
        """
        base_query: dict = {"username": username, "evaluation_run_name": evaluation_run_name}

        if not run_id:
            latest = self._eval_results.find_one(
                base_query,
                {"run_id": 1},
                sort=[("created_at", DESCENDING)],
            )
            if not latest:
                return []
            run_id = latest["run_id"]

        return list(self._eval_results.find(
            {**base_query, "run_id": run_id},
            {"_id": 0},
        ))

    def list_runs(self, username: str, evaluation_run_name: str) -> list[dict]:
        """
        Return all run_ids for a user/evaluation_run_name with their timestamps, newest first.
        Each entry: {run_id, created_at}
        """
        pipeline = [
            {"$match": {"username": username, "evaluation_run_name": evaluation_run_name}},
            {"$group": {
                "_id": "$run_id",
                "created_at": {"$max": "$created_at"},
                "question_count": {"$sum": 1},
            }},
            {"$sort": {"created_at": DESCENDING}},
            {"$project": {"_id": 0, "run_id": "$_id", "created_at": 1, "question_count": 1}},
        ]
        return list(self._eval_results.aggregate(pipeline))

    # ══════════════════════════════════════════════════════════════════════════
    # Metrics
    # ══════════════════════════════════════════════════════════════════════════

    def upsert_metrics(
        self,
        username: str,
        evaluation_run_name: str,
        run_id: str,
        metrics: dict,
        insights: dict | None = None,
    ) -> None:
        """
        Upsert aggregated benchmark metrics for a run.
        Keyed by (username, evaluation_run_name, run_id) — safe to call multiple times.

        Args:
            username:             owning user
            evaluation_run_name:  e.g. "my-rag-system_strict"
            run_id:               run identifier returned by insert_evaluation_results()
            metrics:              output of MetricsCalculator.compute()
            insights:             optional output of InsightsAgent.generate_dict()
        """
        now = _now()
        self._metrics.update_one(
            {
                "username": username,
                "evaluation_run_name": evaluation_run_name,
                "run_id": run_id,
            },
            {
                "$set": {
                    "username": username,
                    "evaluation_run_name": evaluation_run_name,
                    "run_id": run_id,
                    "alpha": metrics.get("e2e", {}).get("alpha", 0.50),
                    "e2e": metrics.get("e2e", {}),
                    "retriever": metrics.get("retriever", {}),
                    "generator": metrics.get("generator", {}),
                    "attribution": metrics.get("attribution", {}),
                    "insights": insights or {},
                    "updated_at": now,
                },
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )
        print(f"  Metrics upserted for run [{run_id}] — [{username}/{evaluation_run_name}]")

    def load_metrics(
        self,
        username: str,
        evaluation_run_name: str,
        run_id: str | None = None,
    ) -> dict | None:
        """
        Load metrics for a run. If run_id is None, returns the most recent run.
        Returns None if no metrics exist.
        """
        query: dict = {"username": username, "evaluation_run_name": evaluation_run_name}
        if run_id:
            query["run_id"] = run_id
            return self._metrics.find_one(query, {"_id": 0})
        return self._metrics.find_one(query, {"_id": 0}, sort=[("created_at", DESCENDING)])

    def list_metrics_history(self, username: str, evaluation_run_name: str) -> list[dict]:
        """
        Return all metrics records for a user/evaluation_run_name, newest first.
        Useful for displaying a run history in the frontend.
        """
        return list(self._metrics.find(
            {"username": username, "evaluation_run_name": evaluation_run_name},
            {"_id": 0},
            sort=[("created_at", DESCENDING)],
        ))

    def list_rag_systems(self, username: str) -> list[str]:
        """Return all distinct rag_system_names registered under a username (from chunk_embeddings)."""
        return self._chunks.distinct("rag_system_name", {"username": username})

    # ══════════════════════════════════════════════════════════════════════════
    # Vector Search
    # ══════════════════════════════════════════════════════════════════════════

    def vector_search_chunks(
        self,
        username: str,
        rag_system_name: str,
        query_embedding: list[float],
        top_k: int = 10,
        num_candidates: int | None = None,
    ) -> list[dict]:
        """
        Semantic nearest-neighbour search via Atlas Vector Search.

        Requires a vector search index named 'embedding_index' on the
        chunk_embeddings collection with:
            path: embedding, numDimensions: 384, similarity: cosine
            filter fields: username, rag_system_name

        Args:
            username:        owning user — used as a server-side filter
            rag_system_name: RAG system name — used as a server-side filter
            query_embedding: 384-dim query vector (list[float])
            top_k:           number of results to return
            num_candidates:  Atlas ANN candidate pool size; defaults to
                             max(top_k * 10, 100) per Atlas recommendation

        Returns:
            list of {source, chunk_id, text, score} dicts, ranked by cosine similarity
        """
        candidates = num_candidates or max(top_k * 10, 100)
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "embedding_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": candidates,
                    "limit": top_k,
                    "filter": {
                        "username": username,
                        "rag_system_name": rag_system_name,
                    },
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "source": 1,
                    "chunk_id": 1,
                    "text": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        return list(self._chunks.aggregate(pipeline))

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
