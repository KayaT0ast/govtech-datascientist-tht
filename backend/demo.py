"""
demo.py — Two-generator OOKB robustness comparison demo.

Evaluates two RAG systems on the same KB documents and questionnaire:
    System A (strict)  — StrictRAGGenerator: aggressive refusal guardrails
    System B (lenient) — LenientRAGGenerator: minimal refusal, helpfulness-first

Same retriever (HybridRetriever from MongoDB), same questionnaire, same judge.
Only the generator's system prompt differs.

The comparison shows how guardrail level affects E2E-OOKRI and how the α
deployment weight shifts which system "wins" (public-facing vs. internal-use).

Usage:
    python backend/demo.py [--kb-dir ./knowledge-base] [--alpha 0.80] [--reindex]
                           [--regen-questionnaire] [--username caleb] [--rag-system-name my_rag]

    --kb-dir              Directory containing KB documents (default: ./knowledge-base)
    --alpha               Deployment weight α (0.80=public-facing, 0.50=balanced, 0.25=internal)
    --reindex             Force re-processing documents and re-embedding into MongoDB
    --regen-questionnaire Delete and regenerate the questionnaire (use after format changes)
    --username            MongoDB owner username (default: USERNAME env var)
    --rag-system-name     RAG system identifier for MongoDB storage (default: kb-dir folder name)

Outputs:
    demo_results.json  — full results, metrics, and insights for both systems
    MongoDB            — chunks+embeddings, questionnaire, evaluation results, and
                         metrics persisted under (username, rag_system_name)

Each pipeline step is skipped if its data already exists in MongoDB:
    chunks        → skipped if chunk_embeddings has entries for this user/system
    questionnaire → skipped if questionnaire collection has entries, or local file exists
    evaluation    → always runs (each call creates a new timestamped run_id)

Requires:
    pip install pymongo sentence-transformers langgraph langchain-huggingface
    MongoDB running at MONGODB_URI in .env
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from scipy.sparse import csr_matrix
from sentence_transformers import SentenceTransformer

# Hyphenated filenames cannot be imported with standard `import` syntax.
import importlib.util

_BACKEND = Path(__file__).parent


def _load(filename: str):
    spec = importlib.util.spec_from_file_location(filename, _BACKEND / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {filename}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_doc       = _load("document-processing.py")
_qgen      = _load("questionnaire-generation.py")
_rag       = _load("rag-agent.py")
_evaluator = _load("evaluator-agent.py")
_insights  = _load("insights-agent.py")
_db_ops    = _load("db_operations.py")

DocumentProcessor       = _doc.DocumentProcessor
QuestionGenerationAgent = _qgen.QuestionGenerationAgent
E2ERAGAgent             = _rag.E2ERAGAgent
RetrieverAgent          = _rag.RetrieverAgent
GeneratorAgent          = _rag.GeneratorAgent
StrictRAGGenerator      = _rag.StrictRAGGenerator
LenientRAGGenerator     = _rag.LenientRAGGenerator
JudgeAgent              = _evaluator.JudgeAgent
MetricsCalculator       = _evaluator.MetricsCalculator
InsightsAgent           = _insights.InsightsAgent
DatabaseManager         = _db_ops.DatabaseManager

load_dotenv()


# ── MongoDB-backed HybridRetriever ────────────────────────────────────────────

class MongoBackedHybridRetriever:
    """
    HybridRetriever that loads pre-computed embeddings from MongoDB via
    DatabaseManager.load_chunk_embeddings().

    TF-IDF is built in-memory from the loaded texts.
    Embeddings are stored as a numpy array after loading.
    """

    def __init__(
        self,
        db: DatabaseManager,
        username: str,
        rag_system_name: str,
        alpha: float = 0.5,
    ):
        from sklearn.feature_extraction.text import TfidfVectorizer

        docs = db.load_chunk_embeddings(username, rag_system_name)
        if not docs:
            raise ValueError(
                f"No chunks found for [{username}/{rag_system_name}]. "
                "Run with --reindex first."
            )

        self.docs = docs
        self.alpha = alpha
        self.texts = [d["text"] for d in docs]

        self.tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = csr_matrix(self.tfidf.fit_transform(self.texts))

        self.embeddings = np.array([d["embedding"] for d in docs], dtype=np.float32)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        encoder = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))

        query_tfidf = csr_matrix(self.tfidf.transform([query]))
        query_vector = query_tfidf.toarray().ravel()
        tfidf_scores = self.tfidf_matrix.dot(query_vector)
        if tfidf_scores.max() > 0:
            tfidf_scores = tfidf_scores / tfidf_scores.max()

        query_emb = encoder.encode([query], convert_to_numpy=True)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norm_embs = self.embeddings / np.where(norms == 0, 1, norms)
        norm_q = query_emb / np.linalg.norm(query_emb)
        semantic_scores = (np.dot(norm_embs, norm_q.T).flatten() + 1) / 2

        scores = self.alpha * tfidf_scores + (1 - self.alpha) * semantic_scores
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "id":      f"{self.docs[i]['source']}_chunk_{self.docs[i]['chunk_id']}",
                "title":   self.docs[i]["source"],
                "content": self.docs[i]["text"],
                "score":   float(scores[i]),
            }
            for i in top_indices
        ]


# ── Pipeline helpers ──────────────────────────────────────────────────────────

def _ensure_chunks(
    db: DatabaseManager,
    username: str,
    rag_system_name: str,
    kb_dir: Path,
    reindex: bool,
) -> None:
    """Process KB documents and upsert chunks+embeddings to MongoDB. Skips if already done."""
    if reindex:
        print("  Re-indexing: clearing existing chunks...")
        db.delete_chunk_embeddings(username, rag_system_name)

    count = db.count_chunk_embeddings(username, rag_system_name)
    if count > 0:
        print(f"  {count} chunks already in MongoDB for [{username}/{rag_system_name}] — skipping.")
        return

    print(f"  Processing documents in {kb_dir}...")
    result = DocumentProcessor().run(kb_dir)
    chunks = result["chunks"]
    print(f"  Processed: {result['processed']}, total chunks: {len(chunks)}")


    encoder = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    db.upsert_chunk_embeddings(username, rag_system_name, chunks, encoder)


def _ensure_questionnaire(
    db: DatabaseManager,
    username: str,
    rag_system_name: str,
    kb_dir: Path,
    regen: bool = False,
) -> list[dict]:
    """
    Return the questionnaire for this user/system.
    Priority: MongoDB → local questionnaire.json → generate fresh.
    Upserts to MongoDB whenever data comes from file or is freshly generated.
    Pass regen=True to force regeneration (needed after questionnaire format changes).
    """
    if regen:
        print("  Clearing existing questionnaire for regeneration...")
        db.delete_questionnaire(username, rag_system_name)
        questionnaire_path = kb_dir / "questionnaire.json"
        if questionnaire_path.exists():
            questionnaire_path.unlink()
            print(f"  Deleted local {questionnaire_path.name}")

    # 1. Check MongoDB — preferred, skip generation entirely
    questions = db.load_questionnaire(username, rag_system_name)
    if questions:
        print(f"  Loaded {len(questions)} questions from MongoDB.")
        return questions

    # 2. Local file exists — load and sync to MongoDB
    questionnaire_path = kb_dir / "questionnaire.json"
    if questionnaire_path.exists():
        print(f"  Loading questionnaire from {questionnaire_path} and syncing to MongoDB...")
        questions = json.loads(questionnaire_path.read_text(encoding="utf-8"))
        db.upsert_questionnaire(username, rag_system_name, questions)
        return questions

    # 3. Generate from scratch
    print("  Generating questionnaire (this may take several minutes)...")
    chunks_path = kb_dir / "kb_chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"kb_chunks.json not found at {chunks_path}. "
            "DocumentProcessor saves it automatically — run with --reindex first."
        )
    agent = QuestionGenerationAgent()
    questions = agent.run(chunks_path=chunks_path, output_path=questionnaire_path, scale=40)
    print(f"  Generated {len(questions)} questions.")
    db.upsert_questionnaire(username, rag_system_name, questions)
    return questions


def _run_system(
    name: str,
    retriever: MongoBackedHybridRetriever,
    generator,
    questions: list[dict],
    judge: JudgeAgent,
    alpha: float,
    db: DatabaseManager,
    username: str,
    rag_system_name: str,
    mode: str,
) -> dict:
    print(f"\n{'='*60}")
    print(f"  Running System {name}  (system={mode})")
    print(f"{'='*60}")

    # Mode 1: E2E — natural retrieval + generation
    e2e_agent = E2ERAGAgent(retriever=retriever, generator=generator)
    e2e_raw = e2e_agent.run_questionnaire(questions)
    print(f"  E2E: queried {len(e2e_raw)} questions.")

    # Mode 2: Retriever-only — measures retrieval quality independently of the generator
    retriever_agent = RetrieverAgent(retriever=retriever)
    retriever_results = retriever_agent.run_questionnaire(questions)
    print(f"  Retriever: ran {len(retriever_results)} queries.")

    # Mode 3: Generator-only — forced context (correct for IKB, empty for OOKB A/B/C, near-miss for D/E)
    gen_agent = GeneratorAgent(generator=generator)
    gen_raw = gen_agent.run_questionnaire(questions)
    print(f"  Generator (forced context): queried {len(gen_raw)} questions.")

    # Judge E2E and generator results — retriever results need no judging (only IDs matter)
    print("  Judging E2E responses...")
    e2e_judged = judge.evaluate_results(e2e_raw)

    print("  Judging generator (forced context) responses...")
    gen_judged = judge.evaluate_results(gen_raw)

    # Combine: MetricsCalculator.compute() splits by mode field ("e2e"/"retriever"/"generator")
    all_results = e2e_judged + retriever_results + gen_judged
    metrics = MetricsCalculator(alpha=alpha).compute(all_results)

    print("  Generating insights...")
    insights = InsightsAgent().generate_dict(metrics, alpha=alpha)

    run_id = db.insert_evaluation_results(username, rag_system_name, all_results)
    db.upsert_metrics(username, rag_system_name, run_id, metrics, insights)
    print(f"  Persisted to MongoDB — run_id: {run_id}")

    return {"results": all_results, "metrics": metrics, "insights": insights, "run_id": run_id}


# ── Console summary ───────────────────────────────────────────────────────────

def _print_comparison(strict: dict, lenient: dict, alpha: float) -> None:
    sm = strict["metrics"].get("e2e", {})
    lm = lenient["metrics"].get("e2e", {})
    sg = strict["metrics"].get("generator", {})
    lg = lenient["metrics"].get("generator", {})
    sr = strict["metrics"].get("retriever", {})
    lr = lenient["metrics"].get("retriever", {})

    def pct(v):
        return f"{v:.1%}" if isinstance(v, float) else "N/A"

    def score(v):
        return f"{v:.3f}" if isinstance(v, float) else "N/A"

    rows = [
        ("Metric",                      "System A (Strict)",       "System B (Lenient)"),
        ("-" * 30,                      "-" * 18,                  "-" * 18),
        ("── E2E ──",                   "",                        ""),
        ("OOKB Abstention Rate",        pct(sm.get("E2E_AR")),     pct(lm.get("E2E_AR"))),
        ("OOKB Hallucination Rate",     pct(sm.get("E2E_HR")),     pct(lm.get("E2E_HR"))),
        ("IKB Coverage",                pct(sm.get("E2E_COV")),    pct(lm.get("E2E_COV"))),
        ("IKB Hallucination Rate",      pct(sm.get("E2E_IK_HR")), pct(lm.get("E2E_IK_HR"))),
        ("IKB Refusal Rate",            pct(sm.get("E2E_IK_RR")), pct(lm.get("E2E_IK_RR"))),
        ("── Retriever ──",             "",                        ""),
        ("E2E Recall@k",                pct(sm.get("R_Rec_k")),    pct(lm.get("R_Rec_k"))),
        ("Standalone Recall@k",         pct(sr.get("R_Rec_k")),    pct(lr.get("R_Rec_k"))),
        ("Standalone Precision@k",      pct(sr.get("R_Prec_k")),   pct(lr.get("R_Prec_k"))),
        ("── Generator (forced ctx) ──","",                        ""),
        ("G-AR  (OOKB refusal)",        pct(sg.get("G_AR")),       pct(lg.get("G_AR"))),
        ("G-FAR (IKB false refusal)",   pct(sg.get("G_FAR")),      pct(lg.get("G_FAR"))),
        ("G-PCCR (near-miss confab.)",  pct(sg.get("G_PCCR")),     pct(lg.get("G_PCCR"))),
        ("-" * 30,                      "-" * 18,                  "-" * 18),
        (f"E2E-OOKRI (α={alpha})",      score(sm.get("E2E_OOKRI")), score(lm.get("E2E_OOKRI"))),
    ]

    print(f"\n{'='*72}")
    print(f"  DEMO RESULTS  |  α = {alpha}  |  "
          f"{'Public-facing' if alpha >= 0.7 else 'Internal-use' if alpha <= 0.3 else 'Balanced'}")
    print(f"{'='*72}")
    for col1, col2, col3 in rows:
        print(f"  {col1:<30}  {col2:<18}  {col3}")
    print(f"{'='*72}")

    si = strict["insights"]
    li = lenient["insights"]
    print(f"\n  System A bottleneck : {si.get('primary_bottleneck')} | "
          f"Safety: {si.get('safety_status')} | Helpfulness: {si.get('helpfulness_status')}")
    print(f"  System B bottleneck : {li.get('primary_bottleneck')} | "
          f"Safety: {li.get('safety_status')} | Helpfulness: {li.get('helpfulness_status')}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OOKB Robustness Demo")
    parser.add_argument("--kb-dir", default="./knowledge-base",
                        help="Directory containing KB documents")
    parser.add_argument("--alpha", type=float, default=0.80,
                        help="Deployment weight α (0.80=public, 0.50=balanced, 0.25=internal)")
    parser.add_argument("--reindex", action="store_true",
                        help="Force re-processing and re-embedding documents")
    parser.add_argument("--regen-questionnaire", action="store_true",
                        help="Delete and regenerate the questionnaire (needed after questionnaire format changes)")
    parser.add_argument("--username", default=None,
                        help="MongoDB owner username (default: USERNAME env var)")
    parser.add_argument("--rag-system-name", default=None,
                        help="RAG system identifier for MongoDB (default: kb-dir folder name)")
    args = parser.parse_args()

    kb_dir = Path(args.kb_dir)
    if not kb_dir.is_dir():
        print(f"Error: KB directory not found: {kb_dir}")
        sys.exit(1)

    username = args.username or os.getenv("USERNAME", "demo_user")
    rag_system_name = args.rag_system_name or kb_dir.resolve().name
    hf_token = os.getenv("HF_TOKEN", "")

    print(f"\nOOKB Demo — user={username!r}  system={rag_system_name!r}  α={args.alpha}")

    db = DatabaseManager()
    try:
        # Step 1 — ingest KB documents into MongoDB (skip if already done)
        print("\n[1/5] Ingesting KB documents...")
        _ensure_chunks(db, username, rag_system_name, kb_dir, reindex=args.reindex)

        # Step 2 — build retriever from MongoDB embeddings
        print("\n[2/5] Building retriever from MongoDB...")
        retriever = MongoBackedHybridRetriever(db, username, rag_system_name)
        print(f"  Retriever ready: {len(retriever.docs)} chunks")

        # Step 3 — load or generate questionnaire (skip if already in MongoDB)
        print("\n[3/5] Preparing questionnaire...")
        questions = _ensure_questionnaire(db, username, rag_system_name, kb_dir,
                                           regen=args.regen_questionnaire)
        print(f"  {len(questions)} questions ready.")

        # Steps 4–5 — run both systems and persist results to MongoDB
        judge = JudgeAgent(hf_token=hf_token)

        print("\n[4/5] Running System A (Strict)...")
        strict_results = _run_system(
            name="A (Strict)",
            retriever=retriever,
            generator=StrictRAGGenerator(hf_token=hf_token),
            questions=questions,
            judge=judge,
            alpha=args.alpha,
            db=db,
            username=username,
            rag_system_name=rag_system_name,
            mode="strict",
        )

        print("\n[5/5] Running System B (Lenient)...")
        lenient_results = _run_system(
            name="B (Lenient)",
            retriever=retriever,
            generator=LenientRAGGenerator(hf_token=hf_token),
            questions=questions,
            judge=judge,
            alpha=args.alpha,
            db=db,
            username=username,
            rag_system_name=rag_system_name,
            mode="lenient",
        )

        # Save full results to JSON
        output = {
            "alpha":           args.alpha,
            "kb_dir":          str(kb_dir),
            "username":        username,
            "rag_system_name": rag_system_name,
            "question_count":  len(questions),
            "strict": {
                "run_id":   strict_results["run_id"],
                "metrics":  strict_results["metrics"],
                "insights": strict_results["insights"],
                "results":  strict_results["results"],
            },
            "lenient": {
                "run_id":   lenient_results["run_id"],
                "metrics":  lenient_results["metrics"],
                "insights": lenient_results["insights"],
                "results":  lenient_results["results"],
            },
        }
        output_path = kb_dir / "demo_results.json"
        output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nFull results saved → {output_path.resolve()}")

    finally:
        db.close()

    _print_comparison(strict_results, lenient_results, args.alpha)


if __name__ == "__main__":
    main()
