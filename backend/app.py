"""
app.py — entry point for the RAG Evaluation Platform.

Responsibilities:
  - Serve the frontend SPA (index.html + static assets)
  - Mount all API routes from api.py
  - Connect to MongoDB on startup

Run:
    py -m uvicorn backend.app:app --reload --port 8000
"""

import logging
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path


class _SuppressJobPolls(logging.Filter):
    """Hides GET /api/jobs/... from the uvicorn access log."""
    def filter(self, record: logging.LogRecord) -> bool:
        return "/api/jobs/" not in record.getMessage()

logging.getLogger("uvicorn.access").addFilter(_SuppressJobPolls())

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

_BACKEND_DIR  = Path(__file__).resolve().parent
_ROOT_DIR     = _BACKEND_DIR.parent
_FRONTEND_DIR = _ROOT_DIR / "frontend"
_STATIC_DIR   = _FRONTEND_DIR / "static"

# Make backend modules importable (db_operations, etc.)
sys.path.insert(0, str(_BACKEND_DIR))

from api import router  # noqa: E402  (after sys.path insert)


# ── Startup ───────────────────────────────────────────────────────────────────

def _prewarm_models() -> None:
    """Load both LLM pipelines in the background so the first request isn't slow."""
    try:
        import local_inference
        print("[startup] Pre-warming LLM models …")
        local_inference._ensure_loaded()
        print("[startup] LLM models ready.")
    except Exception as exc:
        print(f"[startup] WARNING: model pre-warm failed: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eagerly connect to MongoDB so any misconfiguration surfaces at startup
    from api import get_db
    try:
        get_db()
        print("MongoDB connected.")
    except Exception as e:
        print(f"WARNING: MongoDB unavailable. ({e})")

    # Load LLM models in the background; server is available immediately
    threading.Thread(target=_prewarm_models, daemon=True).start()

    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="RAG Evaluation Platform", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(router)

# Static assets  (CSS, JS, images)
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ── SPA fallback — serve index.html for every non-API path ───────────────────

@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    return FileResponse(str(_FRONTEND_DIR / "index.html"))
