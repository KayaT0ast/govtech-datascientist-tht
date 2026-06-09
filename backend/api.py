"""
api.py — FastAPI REST backend for the RAG Evaluation Platform.

Routes
------
Auth:         POST /api/auth/register  POST /api/auth/login
Systems:      GET  /api/systems
KB:           POST /api/kb/upload   GET /api/kb/status   DELETE /api/kb
Questionnaire:POST /api/questionnaire/generate   GET /api/questionnaire
Evaluation:   POST /api/evaluate
Jobs:         GET  /api/jobs/{job_id}
Metrics:      GET  /api/metrics   GET /api/runs   GET /api/results/{run_id}
Insights:     POST /api/insights
Demo:         POST /api/demo/query   /api/demo/retrieve   /api/demo/generate
Static:       /* → frontend/index.html

Run:
    cd "GovTech TAP THA"
    uvicorn backend.api:app --reload --port 8000
"""

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Header, HTTPException, UploadFile
from jose import JWTError, jwt
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────

_BACKEND   = Path(__file__).parent
_ROOT      = _BACKEND.parent
_FRONTEND  = _ROOT / "frontend"
_KB_BASE   = _ROOT / "knowledge-base"
_EVAL_BASE = _ROOT / "evaluation-results"

# ── Module loader (hyphenated filenames) ──────────────────────────────────────

def _load(filename: str):
    # Use a valid Python identifier as the sys.modules key so re-imports
    # of this file (e.g. from a different agent) return the cached module.
    key = filename.removesuffix(".py").replace("-", "_")
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, _BACKEND / filename)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod          # register BEFORE exec to allow circular imports
    spec.loader.exec_module(mod)
    return mod

_doc       = _load("document-processing.py")
_qgen_mod  = _load("questionnaire-generation.py")
_rag       = _load("rag-agent.py")
_eval_mod  = _load("evaluator-agent.py")
_ins_mod   = _load("insights-agent.py")

sys.path.insert(0, str(_BACKEND))
from db_operations import DatabaseManager  # noqa: E402 (after path insert)

DocumentProcessor       = _doc.DocumentProcessor
QuestionGenerationAgent = _qgen_mod.QuestionGenerationAgent
E2ERAGAgent             = _rag.E2ERAGAgent
RetrieverAgent          = _rag.RetrieverAgent
GeneratorAgent          = _rag.GeneratorAgent
StrictRAGGenerator      = _rag.StrictRAGGenerator
LenientRAGGenerator     = _rag.LenientRAGGenerator
JudgeAgent              = _eval_mod.JudgeAgent
MetricsCalculator       = _eval_mod.MetricsCalculator
InsightsAgent           = _ins_mod.InsightsAgent

# ── Router ────────────────────────────────────────────────────────────────────

router = APIRouter()

# ── Singletons ────────────────────────────────────────────────────────────────

_db: DatabaseManager | None = None
_encoder: SentenceTransformer | None = None
_retriever_cache: dict = {}   # (username, rag_system_name) → retriever instance
_jobs: dict[str, dict] = {}   # job_id → job state

def get_db() -> DatabaseManager:
    global _db
    if _db is None:
        _db = DatabaseManager()
    return _db

def get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    return _encoder

# ── In-memory retriever (loads embeddings from MongoDB) ──────────────────────

class _LocalRetriever:
    """In-memory hybrid retriever backed by MongoDB embeddings."""

    def __init__(self, db: DatabaseManager, username: str, rag_system_name: str, alpha: float = 0.5):
        docs = db.load_chunk_embeddings(username, rag_system_name)
        if not docs:
            raise ValueError(f"No chunks for [{username}/{rag_system_name}]")
        self.docs = docs
        self.alpha = alpha
        self.texts = [d["text"] for d in docs]
        self.tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.tfidf.fit_transform(self.texts)
        self.embeddings = np.array([d["embedding"] for d in docs], dtype=np.float32)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        encoder = get_encoder()
        q_tfidf = self.tfidf.transform([query])
        tfidf_scores = (self.tfidf_matrix * q_tfidf.T).toarray().flatten()
        if tfidf_scores.max() > 0:
            tfidf_scores = tfidf_scores / tfidf_scores.max()

        q_emb = encoder.encode([query], convert_to_numpy=True)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norm_embs = self.embeddings / np.where(norms == 0, 1, norms)
        norm_q = q_emb / np.linalg.norm(q_emb)
        sem_scores = (np.dot(norm_embs, norm_q.T).flatten() + 1) / 2

        scores = self.alpha * tfidf_scores + (1 - self.alpha) * sem_scores
        top_idx = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "id":      f"{self.docs[i]['source']}_chunk_{self.docs[i]['chunk_id']}",
                "title":   self.docs[i]["source"],
                "content": self.docs[i]["text"],
                "score":   float(scores[i]),
            }
            for i in top_idx
        ]


def _get_retriever(username: str, rag_system_name: str) -> _LocalRetriever:
    key = (username, rag_system_name)
    if key not in _retriever_cache:
        _retriever_cache[key] = _LocalRetriever(get_db(), username, rag_system_name)
    return _retriever_cache[key]

def _invalidate_retriever(username: str, rag_system_name: str) -> None:
    _retriever_cache.pop((username, rag_system_name), None)

# ── Auth (JWT) ────────────────────────────────────────────────────────────────

_SECRET   = os.getenv("JWT_SECRET", "govtech-rag-secret-key-change-in-prod")
_ALGORITHM = "HS256"
_TOKEN_HOURS = 24 * 7


def _make_token(username: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(hours=_TOKEN_HOURS)
    return jwt.encode({"sub": username, "exp": exp}, _SECRET, algorithm=_ALGORITHM)


async def _current_user(authorization: Optional[str] = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    try:
        payload = jwt.decode(authorization[7:], _SECRET, algorithms=[_ALGORITHM])
        username: str = payload.get("sub", "")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Token expired or invalid")

# ── Job helpers ───────────────────────────────────────────────────────────────

def _new_job() -> str:
    jid = f"job_{uuid.uuid4().hex[:12]}"
    _jobs[jid] = {"status": "pending", "step": "queued", "progress": 0, "message": "Job queued"}
    return jid

def _update_job(jid: str, **kwargs) -> None:
    if jid in _jobs:
        _jobs[jid].update(kwargs)

def _finish_job(jid: str, result: dict) -> None:
    _update_job(jid, status="done", progress=100, step="done", message="Completed", result=result)

def _fail_job(jid: str, error: str) -> None:
    _update_job(jid, status="failed", step="error", message=error, error=error)

# ── KB directory helper ───────────────────────────────────────────────────────

def _kb_dir(username: str, rag_system_name: str) -> Path:
    d = _KB_BASE / username / rag_system_name
    d.mkdir(parents=True, exist_ok=True)
    return d

def _eval_dir(username: str) -> Path:
    d = _EVAL_BASE / username
    d.mkdir(parents=True, exist_ok=True)
    return d

# ── Pydantic models ───────────────────────────────────────────────────────────

class AuthRequest(BaseModel):
    username: str
    password: str

class QuestionnaireGenRequest(BaseModel):
    rag_system_name: str
    scale: int = 40
    regen: bool = False

class EvaluateRequest(BaseModel):
    rag_system_name: str
    alpha: float = 0.80
    e2e_url: Optional[str] = None
    retriever_url: Optional[str] = None
    generator_url: Optional[str] = None
    regen_questionnaire: bool = False

class InsightsRequest(BaseModel):
    metrics: dict
    alpha: float = 0.80

class DemoQueryRequest(BaseModel):
    question: str
    rag_system_name: str
    generator: str = "strict"   # "strict" | "lenient"

class DemoRetrieveRequest(BaseModel):
    question: str
    rag_system_name: str

class DemoGenerateRequest(BaseModel):
    question: str
    context: list
    generator: str = "strict"

class KBDeleteRequest(BaseModel):
    rag_system_name: str

# ══════════════════════════════════════════════════════════════════════════════
# Auth endpoints
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/api/auth/register")
async def register(req: AuthRequest):
    db = get_db()
    if not req.username.strip() or not req.password:
        raise HTTPException(400, "Username and password required")
    ok = db.register_user(req.username.strip(), req.password)
    if not ok:
        raise HTTPException(409, "Username already taken")
    return {"token": _make_token(req.username), "username": req.username}


@router.post("/api/auth/login")
async def login(req: AuthRequest):
    db = get_db()
    if not db.verify_user(req.username, req.password):
        raise HTTPException(401, "Invalid credentials")
    return {"token": _make_token(req.username), "username": req.username}

# ══════════════════════════════════════════════════════════════════════════════
# RAG systems
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/api/systems")
async def list_systems(username: str = Depends(_current_user)):
    db = get_db()
    # list_rag_systems queries chunk_embeddings which stores bare rag_system_names
    systems = db.list_rag_systems(username)
    return {"systems": sorted(systems)}

# ══════════════════════════════════════════════════════════════════════════════
# Knowledge base
# ══════════════════════════════════════════════════════════════════════════════

def _bg_upload(
    jid: str,
    username: str,
    rag_system_name: str,
    tmp_paths: list[tuple[str, str]],   # (tmp_path, original_filename)
    reindex: bool,
    scale: int = 40,
) -> None:
    db = get_db()
    encoder = get_encoder()
    try:
        _update_job(jid, status="processing", step="processing_docs", progress=5,
                    message="Processing documents…")
        kb = _kb_dir(username, rag_system_name)

        for tmp_path, original_name in tmp_paths:
            shutil.move(tmp_path, kb / original_name)

        if reindex:
            db.delete_chunk_embeddings(username, rag_system_name)
            _invalidate_retriever(username, rag_system_name)
            chunks_path = kb / "kb_chunks.json"
            if chunks_path.exists():
                chunks_path.unlink()
            db.delete_questionnaire(username, rag_system_name)
            q_path = kb / "questionnaire.json"
            if q_path.exists():
                q_path.unlink()

        _update_job(jid, step="chunking", progress=15, message="Chunking documents…")
        result = DocumentProcessor().run(kb)
        chunks = result["chunks"]

        _update_job(jid, step="embedding", progress=35, message=f"Embedding {len(chunks)} chunks…")
        db.upsert_chunk_embeddings(username, rag_system_name, chunks, encoder)
        _invalidate_retriever(username, rag_system_name)

        # ── Questionnaire generation ──────────────────────────────────────────
        _update_job(jid, step="generating_questionnaire", progress=50,
                    message="Generating questionnaire (this may take a few minutes)…")

        existing = db.load_questionnaire(username, rag_system_name)
        if existing and not reindex:
            question_count = len(existing)
            q_source = "cache"
        else:
            chunks_path = kb / "kb_chunks.json"
            agent = QuestionGenerationAgent()

            def _q_progress(message: str, progress: int):
                _update_job(jid, step="generating_questionnaire",
                            progress=progress, message=message)

            questions = agent.run(
                chunks_path=chunks_path,
                output_path=kb / "questionnaire.json",
                scale=scale,
                progress_callback=_q_progress,
            )
            _update_job(jid, step="saving_questionnaire", progress=90,
                        message=f"Saving {len(questions)} questions…")
            db.upsert_questionnaire(username, rag_system_name, questions)
            question_count = len(questions)
            q_source = "generated"

        _finish_job(jid, {
            "chunk_count": len(chunks),
            "processed": result["processed"],
            "skipped": result["skipped"],
            "question_count": question_count,
            "question_source": q_source,
        })
    except Exception as e:
        _fail_job(jid, str(e))
    finally:
        for tmp_path, _ in tmp_paths:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass


@router.post("/api/kb/upload")
async def kb_upload(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    rag_system_name: str = Form(...),
    reindex: bool = Form(False),
    scale: int = Form(40),
    username: str = Depends(_current_user),
):
    if not rag_system_name.strip():
        raise HTTPException(400, "rag_system_name required")

    tmp_paths: list[tuple[str, str]] = []
    for f in files:
        original_name = Path(f.filename or "file").name
        suffix = Path(original_name).suffix
        fd, tmp = tempfile.mkstemp(suffix=suffix, prefix="kb_")
        os.close(fd)
        Path(tmp).write_bytes(await f.read())
        tmp_paths.append((tmp, original_name))

    if not tmp_paths:
        raise HTTPException(400, "No files uploaded")

    jid = _new_job()
    background_tasks.add_task(_bg_upload, jid, username, rag_system_name.strip(), tmp_paths, reindex, scale)
    return {"job_id": jid}


@router.get("/api/kb/status")
async def kb_status(rag_system_name: str, username: str = Depends(_current_user)):
    db = get_db()
    count = db.count_chunk_embeddings(username, rag_system_name)
    return {"rag_system_name": rag_system_name, "chunk_count": count}


@router.delete("/api/kb")
async def kb_delete(req: KBDeleteRequest, username: str = Depends(_current_user)):
    db = get_db()
    deleted = db.delete_chunk_embeddings(username, req.rag_system_name)
    _invalidate_retriever(username, req.rag_system_name)
    return {"deleted": deleted}

# ══════════════════════════════════════════════════════════════════════════════
# Questionnaire
# ══════════════════════════════════════════════════════════════════════════════

def _bg_questionnaire(
    jid: str,
    username: str,
    rag_system_name: str,
    scale: int,
    regen: bool,
) -> None:
    db = get_db()
    try:
        _update_job(jid, status="processing", step="preparing", progress=5,
                    message="Preparing questionnaire generation…")
        kb = _kb_dir(username, rag_system_name)

        if regen:
            db.delete_questionnaire(username, rag_system_name)
            q_path = kb / "questionnaire.json"
            if q_path.exists():
                q_path.unlink()

        # Check MongoDB first
        existing = db.load_questionnaire(username, rag_system_name)
        if existing and not regen:
            _finish_job(jid, {"question_count": len(existing), "source": "cache"})
            return

        chunks_path = kb / "kb_chunks.json"
        if not chunks_path.exists():
            raise FileNotFoundError(
                "kb_chunks.json not found — upload KB documents first."
            )

        _update_job(jid, step="generating", progress=20, message="Generating questionnaire (this may take a few minutes)…")
        agent = QuestionGenerationAgent()
        questions = agent.run(
            chunks_path=chunks_path,
            output_path=kb / "questionnaire.json",
            scale=scale,
        )

        _update_job(jid, step="saving", progress=90, message=f"Saving {len(questions)} questions to MongoDB…")
        db.upsert_questionnaire(username, rag_system_name, questions)

        _finish_job(jid, {"question_count": len(questions), "source": "generated"})
    except Exception as e:
        _fail_job(jid, str(e))


@router.post("/api/questionnaire/generate")
async def questionnaire_generate(
    req: QuestionnaireGenRequest,
    background_tasks: BackgroundTasks,
    username: str = Depends(_current_user),
):
    jid = _new_job()
    background_tasks.add_task(
        _bg_questionnaire, jid, username, req.rag_system_name, req.scale, req.regen
    )
    return {"job_id": jid}


@router.get("/api/questionnaire")
async def questionnaire_get(rag_system_name: str, username: str = Depends(_current_user)):
    db = get_db()
    questions = db.load_questionnaire(username, rag_system_name)
    return {"questions": questions, "count": len(questions)}


class QuestionUpdateRequest(BaseModel):
    rag_system_name: str
    question: Optional[str] = None
    category: Optional[str] = None
    type: Optional[str] = None
    difficulty: Optional[str] = None
    expected_behavior: Optional[str] = None
    expected_answer: Optional[str] = None
    source_documents: Optional[list] = None


@router.put("/api/questionnaire/{question_id}")
async def questionnaire_update(
    question_id: str,
    req: QuestionUpdateRequest,
    username: str = Depends(_current_user),
):
    db = get_db()
    questions = db.load_questionnaire(username, req.rag_system_name)
    match = next((q for q in questions if q["id"] == question_id), None)
    if not match:
        raise HTTPException(404, "Question not found")

    # Merge editable fields over the existing question
    fields = ["question", "category", "type", "difficulty",
              "expected_behavior", "expected_answer", "source_documents"]
    for f in fields:
        val = getattr(req, f)
        if val is not None:
            match[f] = val

    db.upsert_questionnaire(username, req.rag_system_name, [match])
    return {"question": match}

# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def _run_one_system(
    system_label: str,
    retriever,
    generator,
    external_url: str | None,
    retriever_url: str | None,
    generator_url: str | None,
    questions: list[dict],
    judge: JudgeAgent,
    alpha: float,
    db: DatabaseManager,
    username: str,
    evaluation_run_name: str,
    jid: str,
    progress_start: int,
    progress_end: int,
) -> dict:
    """Run E2E + Retriever + Generator evaluation for one system and persist results."""
    span = progress_end - progress_start

    _update_job(jid, step=f"e2e_{system_label}", progress=progress_start + span * 0 // 5,
                message=f"[{system_label}] Running E2E evaluation…")

    # E2E
    if external_url and system_label == "external":
        e2e_agent = E2ERAGAgent(external_url=external_url)
    else:
        e2e_agent = E2ERAGAgent(retriever=retriever, generator=generator)
    e2e_raw = e2e_agent.run_questionnaire(questions)

    _update_job(jid, step=f"retriever_{system_label}", progress=progress_start + span * 1 // 5,
                message=f"[{system_label}] Running retriever evaluation…")

    # Retriever
    if retriever_url and system_label == "external":
        ret_agent = RetrieverAgent(retrieve_url=retriever_url)
    else:
        ret_agent = RetrieverAgent(retriever=retriever)
    ret_results = ret_agent.run_questionnaire(questions)

    _update_job(jid, step=f"generator_{system_label}", progress=progress_start + span * 2 // 5,
                message=f"[{system_label}] Running generator evaluation…")

    # Generator (forced context)
    if generator_url and system_label == "external":
        gen_agent = GeneratorAgent(generate_url=generator_url)
    else:
        gen_agent = GeneratorAgent(generator=generator)
    gen_raw = gen_agent.run_questionnaire(questions)

    _update_job(jid, step=f"judging_{system_label}", progress=progress_start + span * 3 // 5,
                message=f"[{system_label}] Judging responses…")

    # Judge
    e2e_judged = judge.evaluate_results(e2e_raw)
    gen_judged  = judge.evaluate_results(gen_raw)

    _update_job(jid, step=f"metrics_{system_label}", progress=progress_start + span * 4 // 5,
                message=f"[{system_label}] Computing metrics…")

    all_results = e2e_judged + ret_results + gen_judged
    metrics     = MetricsCalculator(alpha=alpha).compute(all_results)
    insights    = InsightsAgent().generate_dict(metrics, alpha=alpha)

    run_id = db.insert_evaluation_results(username, evaluation_run_name, all_results)
    db.upsert_metrics(username, evaluation_run_name, run_id, metrics, insights)

    # Save result file to evaluation-results/<username>/
    try:
        eval_dir = _eval_dir(username)
        result_file = eval_dir / f"{evaluation_run_name}_{run_id}.json"
        result_file.write_text(
            json.dumps({"evaluation_run_name": evaluation_run_name, "run_id": run_id,
                        "metrics": metrics, "insights": insights, "results": all_results},
                       default=str),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"  Warning: could not write eval result file: {e}")

    return {"run_id": run_id, "metrics": metrics, "insights": insights}


def _bg_evaluate(
    jid: str,
    username: str,
    rag_system_name: str,
    alpha: float,
    e2e_url: str | None,
    retriever_url: str | None,
    generator_url: str | None,
    regen_questionnaire: bool,
) -> None:
    db = get_db()
    try:
        _update_job(jid, status="processing", step="loading", progress=2,
                    message="Loading questionnaire…")

        kb = _kb_dir(username, rag_system_name)
        if regen_questionnaire:
            db.delete_questionnaire(username, rag_system_name)
            q_path = kb / "questionnaire.json"
            if q_path.exists():
                q_path.unlink()

        questions = db.load_questionnaire(username, rag_system_name)
        if not questions:
            raise ValueError(
                "No questionnaire found. Generate one first via /api/questionnaire/generate."
            )

        _update_job(jid, step="loading_retriever", progress=5,
                    message=f"Loading retriever ({len(questions)} questions)…")

        retriever = _get_retriever(username, rag_system_name)
        judge     = JudgeAgent()

        strict_gen  = StrictRAGGenerator()
        lenient_gen = LenientRAGGenerator()

        results: dict[str, dict] = {}

        # System A — Strict
        results["strict"] = _run_one_system(
            "strict", retriever, strict_gen,
            None, None, None,
            questions, judge, alpha, db, username,
            f"{rag_system_name}_strict",
            jid, 10, 40,
        )

        # System B — Lenient
        results["lenient"] = _run_one_system(
            "lenient", retriever, lenient_gen,
            None, None, None,
            questions, judge, alpha, db, username,
            f"{rag_system_name}_lenient",
            jid, 40, 70,
        )

        # System C — External (only if URL provided)
        if e2e_url:
            results["external"] = _run_one_system(
                "external", retriever, lenient_gen,
                e2e_url, retriever_url, generator_url,
                questions, judge, alpha, db, username,
                f"{rag_system_name}_external",
                jid, 70, 98,
            )

        _finish_job(jid, results)
    except Exception as e:
        _fail_job(jid, str(e))


@router.post("/api/evaluate")
async def evaluate(
    req: EvaluateRequest,
    background_tasks: BackgroundTasks,
    username: str = Depends(_current_user),
):
    jid = _new_job()
    background_tasks.add_task(
        _bg_evaluate,
        jid, username, req.rag_system_name, req.alpha,
        req.e2e_url, req.retriever_url, req.generator_url,
        req.regen_questionnaire,
    )
    return {"job_id": jid}

# ══════════════════════════════════════════════════════════════════════════════
# Job status
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/api/jobs/{job_id}")
async def job_status(job_id: str, username: str = Depends(_current_user)):
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    return {"job_id": job_id, **_jobs[job_id]}

# ══════════════════════════════════════════════════════════════════════════════
# Metrics & results
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/api/metrics")
async def get_metrics(
    rag_system_name: str,
    run_id: Optional[str] = None,
    system: Optional[str] = None,
    username: str = Depends(_current_user),
):
    """
    Return metrics for a rag_system_name.
    `system` param selects "strict" | "lenient" | "external" (defaults to "strict").
    Appends the suffix automatically so the frontend can use the base name.
    """
    db = get_db()
    suffix = f"_{system}" if system else "_strict"
    evaluation_run_name = rag_system_name + suffix
    m = db.load_metrics(username, evaluation_run_name, run_id)
    if not m:
        return {"metrics": None}
    m.pop("_id", None)
    return {"metrics": m}


@router.get("/api/runs")
async def list_runs(
    rag_system_name: str,
    system: Optional[str] = None,
    username: str = Depends(_current_user),
):
    db = get_db()
    suffix = f"_{system}" if system else "_strict"
    evaluation_run_name = rag_system_name + suffix
    runs = db.list_runs(username, evaluation_run_name)
    for r in runs:
        if isinstance(r.get("created_at"), datetime):
            r["created_at"] = r["created_at"].isoformat()
    return {"runs": runs}


@router.get("/api/results/{run_id}")
async def get_results(
    run_id: str,
    rag_system_name: str,
    system: Optional[str] = None,
    username: str = Depends(_current_user),
):
    db = get_db()
    suffix = f"_{system}" if system else "_strict"
    evaluation_run_name = rag_system_name + suffix
    results = db.load_evaluation_results(username, evaluation_run_name, run_id)
    for r in results:
        r.pop("_id", None)
        if isinstance(r.get("created_at"), datetime):
            r["created_at"] = r["created_at"].isoformat()
    return {"results": results, "count": len(results)}

# ══════════════════════════════════════════════════════════════════════════════
# Insights
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/api/insights")
async def generate_insights(req: InsightsRequest, username: str = Depends(_current_user)):
    report = InsightsAgent().generate_dict(req.metrics, alpha=req.alpha)
    return report

# ══════════════════════════════════════════════════════════════════════════════
# Demo endpoints
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/api/demo/query")
async def demo_query(req: DemoQueryRequest, username: str = Depends(_current_user)):
    try:
        retriever = _get_retriever(username, req.rag_system_name)
    except ValueError as e:
        raise HTTPException(400, str(e))

    gen_cls = StrictRAGGenerator if req.generator == "strict" else LenientRAGGenerator
    generator = gen_cls()
    hits = retriever.retrieve(req.question, top_k=3)
    response = generator.generate(req.question, hits)
    return {
        "response": response,
        "retrieved_contexts": [h.get("content", "") for h in hits],
        "retrieved_ids": [h.get("id", "") for h in hits],
    }


@router.post("/api/demo/retrieve")
async def demo_retrieve(req: DemoRetrieveRequest, username: str = Depends(_current_user)):
    try:
        retriever = _get_retriever(username, req.rag_system_name)
    except ValueError as e:
        raise HTTPException(400, str(e))

    hits = retriever.retrieve(req.question, top_k=5)
    return {
        "chunks": [
            {"id": h.get("id", ""), "content": h.get("content", ""), "score": h.get("score", 0.0)}
            for h in hits
        ]
    }


@router.post("/api/demo/generate")
async def demo_generate(req: DemoGenerateRequest, username: str = Depends(_current_user)):
    gen_cls = StrictRAGGenerator if req.generator == "strict" else LenientRAGGenerator
    generator = gen_cls()
    response = generator.generate(req.question, req.context)
    return {"response": response}

# ══════════════════════════════════════════════════════════════════════════════
# Public RAG endpoints  (no auth — called by the evaluator as external system)
#
# Enter these in the Evaluate page:
#   E2E URL:       http://localhost:8000/rag/<system_name>/query
#   Retriever URL: http://localhost:8000/rag/<system_name>/retrieve
#   Generator URL: http://localhost:8000/rag/<system_name>/generate
#
# <system_name> must match the rag_system_name you uploaded your KB under.
# The owner username is read from the USERNAME env var.
# ══════════════════════════════════════════════════════════════════════════════

def _rag_username() -> str:
    """Owner username for the public RAG endpoints — reads USERNAME from .env."""
    return os.getenv("USERNAME", "")


class _RagQueryRequest(BaseModel):
    query: str
    generator: str = "strict"   # "strict" | "lenient"


class _RagRetrieveRequest(BaseModel):
    query: str
    top_k: int = 3


class _RagGenerateRequest(BaseModel):
    query: str
    context: list = []
    generator: str = "strict"   # "strict" | "lenient"


@router.post("/rag/{rag_system_name}/query")
async def rag_query(rag_system_name: str, req: _RagQueryRequest):
    """
    Full E2E RAG endpoint.  Returns the same shape the evaluator expects:
        {"response": "...", "retrieved_contexts": [...], "retrieved_ids": [...]}
    """
    username = _rag_username()
    try:
        retriever = _get_retriever(username, rag_system_name)
    except ValueError as e:
        raise HTTPException(400, str(e))

    gen_cls   = StrictRAGGenerator if req.generator == "strict" else LenientRAGGenerator
    generator = gen_cls()
    hits      = retriever.retrieve(req.query, top_k=3)
    response  = generator.generate(req.query, hits)
    return {
        "response":           response,
        "retrieved_contexts": [h.get("content", "") for h in hits],
        "retrieved_ids":      [h.get("id", "")      for h in hits],
    }


@router.post("/rag/{rag_system_name}/retrieve")
async def rag_retrieve(rag_system_name: str, req: _RagRetrieveRequest):
    """
    Retriever-only endpoint.  Returns:
        {"chunks": [{"id": "...", "content": "...", "score": 0.9}, ...]}
    """
    username = _rag_username()
    try:
        retriever = _get_retriever(username, rag_system_name)
    except ValueError as e:
        raise HTTPException(400, str(e))

    hits = retriever.retrieve(req.query, top_k=req.top_k)
    return {
        "chunks": [
            {"id": h.get("id", ""), "content": h.get("content", ""), "score": h.get("score", 0.0)}
            for h in hits
        ]
    }


@router.post("/rag/{rag_system_name}/generate")
async def rag_generate(rag_system_name: str, req: _RagGenerateRequest):
    """
    Generator-only endpoint.  Context is injected by the benchmark (forced context).
    Returns: {"response": "..."}
    """
    gen_cls   = StrictRAGGenerator if req.generator == "strict" else LenientRAGGenerator
    generator = gen_cls()
    response  = generator.generate(req.query, req.context)
    return {"response": response}

