"""
mongo-store.py

MongoDBChunkStore — persists KB text chunks and their embeddings to MongoDB
so the HybridRetriever can be rebuilt across demo runs without re-encoding.

MongoDB schema (one document per chunk):
    {
        "source":    "filename.pdf",
        "chunk_id":  0,
        "text":      "chunk content",
        "embedding": [0.12, -0.34, ...]   # 384-dim all-MiniLM-L6-v2 vector
    }

TF-IDF is NOT stored — it is cheap to recompute in-memory from the loaded
texts and its vocabulary changes whenever chunks change, making storage
fragile. Only the expensive embedding computation is persisted.

Requires:
    pip install pymongo sentence-transformers python-dotenv
    MONGODB_URI, MONGODB_DB, MONGODB_COLLECTION in .env
"""

import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection

load_dotenv()

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class MongoDBChunkStore:
    """
    Thin persistence layer for KB chunks and their embeddings.

    Usage:
        store = MongoDBChunkStore()

        # Index documents (run once, or after KB changes)
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        store.store_chunks(chunks, encoder)

        # Load for retrieval (fast — embeddings already computed)
        docs = store.load_chunks()
        # docs is a list of {source, chunk_id, text, embedding}
        # Pass directly to MongoBackedHybridRetriever or HybridRetriever
    """

    def __init__(
        self,
        uri: str | None = None,
        db_name: str | None = None,
        collection_name: str | None = None,
    ):
        self.uri = uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        self.db_name = db_name or os.getenv("MONGODB_DB", "ookb_benchmark")
        self.collection_name = collection_name or os.getenv("MONGODB_COLLECTION", "kb_chunks")

        self._client = MongoClient(
            self.uri,
            # Atlas load balancer terminates idle connections after ~60s.
            # maxIdleTimeMS=45000 makes the driver close idle pool connections
            # proactively at 45s so stale connections are never reused.
            maxIdleTimeMS=45_000,
            # How long to wait when establishing a new connection or selecting
            # a server. Keeps hangs bounded during transient Atlas disruptions.
            connectTimeoutMS=10_000,
            serverSelectionTimeoutMS=10_000,
            # Retry both reads and writes on network errors / stale connections.
            retryReads=True,
            retryWrites=True,
        )
        self._col: Collection = self._client[self.db_name][self.collection_name]

        # Ensure unique index on (source, chunk_id) for upsert deduplication
        self._col.create_index(
            [("source", 1), ("chunk_id", 1)],
            unique=True,
            background=True,
        )

    # ── Write ─────────────────────────────────────────────────────────────────

    def store_chunks(
        self,
        chunks: list[dict],
        encoder: "SentenceTransformer",
        batch_size: int = 64,
    ) -> int:
        """
        Encode chunks and upsert into MongoDB.

        Args:
            chunks:     list of {source, chunk_id, text} from DocumentProcessingAgent
            encoder:    SentenceTransformer instance (e.g. all-MiniLM-L6-v2)
            batch_size: encoding batch size (tune for available memory)

        Returns:
            number of chunks upserted
        """
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        print(f"  Encoding {len(texts)} chunks...")
        embeddings = encoder.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        ops = []
        for chunk, emb in zip(chunks, embeddings):
            ops.append(UpdateOne(
                filter={"source": chunk["source"], "chunk_id": chunk["chunk_id"]},
                update={"$set": {
                    "source":    chunk["source"],
                    "chunk_id":  chunk["chunk_id"],
                    "text":      chunk["text"],
                    "embedding": emb.tolist(),
                }},
                upsert=True,
            ))

        if ops:
            result = self._col.bulk_write(ops, ordered=False)
            print(f"  Stored: {result.upserted_count} new, {result.modified_count} updated")
            return result.upserted_count + result.modified_count

        return 0

    # ── Read ──────────────────────────────────────────────────────────────────

    def load_chunks(self) -> list[dict]:
        """
        Return all chunks as a list of {source, chunk_id, text, embedding}.
        embedding is a Python list[float] (384 dims for all-MiniLM-L6-v2).
        """
        return [
            {
                "source":    doc["source"],
                "chunk_id":  doc["chunk_id"],
                "text":      doc["text"],
                "embedding": doc["embedding"],
            }
            for doc in self._col.find({}, {"_id": 0})
        ]

    def chunk_count(self) -> int:
        """Return the number of stored chunks."""
        return self._col.count_documents({})

    # ── Maintenance ───────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Drop all documents from the collection (use before re-indexing)."""
        self._col.delete_many({})
        print(f"  Cleared collection '{self.collection_name}'")

    def close(self) -> None:
        """Close the MongoDB connection."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
