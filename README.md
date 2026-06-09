# OOKB Robustness Benchmark — a RAG Evaluation Platform

A self-hostable platform for stress-testing whether a Retrieval-Augmented
Generation (RAG) system **knows what it doesn't know**. You upload a knowledge
base, the platform auto-generates a mixed In-Knowledge-Base / Out-of-Knowledge-Base
(IKB/OOKB) questionnaire, runs the RAG system across three evaluation modes,
judges every response with an LLM, and reports a single deployment-weighted
robustness score plus a diagnosis of *where* the system fails.

> **Demo video:** `[DEMO VIDEO URL — add before submission]`

---

## 1. What & why

**The problem.** RAG assistants are spreading across the Singapore public
sector — over policy booklets, SOPs, scheme eligibility rules, and internal
playbooks. Their most dangerous failure mode is not getting an answer wrong; it
is **confabulating a confident, plausible answer to a question the knowledge
base cannot actually support**. A citizen-facing assistant that invents a
MediShield Life deductible, or an internal tool that fabricates a grant ceiling,
erodes trust far faster than an honest "I don't know."

Existing RAG evaluation tooling mostly measures answer quality on questions the
system *should* be able to answer. It rarely measures the inverse: **abstention
discipline** on questions the system *should refuse*. This project targets that
gap. (It corresponds to Annex A sketch #15, "Out-of-knowledge-base robustness.")

**The stakeholder.** A GovTech / agency engineering team about to ship a RAG
assistant who needs an honest, repeatable answer to: *"Before this goes live,
how often does it make things up — and is the weak link the retriever or the
generator?"*

**Why AI is central.** Remove the LLM and there is no product. The LLM does
three irreplaceable jobs: (1) reads the KB to author questions that are
*plausibly* answerable but actually out-of-scope — the hard part of an OOKB
benchmark; (2) acts as the RAG generator under test; and (3) serves as the
rubric-driven judge that classifies each response. The retrieval, metrics, and
scoring around it are deterministic plumbing.

---

## 2. Architecture

Single FastAPI container serving a vanilla HTML/CSS/JS single-page app, backed by
MongoDB for persistence and local LLM inference via Hugging Face `transformers`.

```
                        ┌────────────────────────────────────────────┐
  Browser SPA  ───────► │  FastAPI (backend/app.py → backend/api.py)   │
  (frontend/)           │   JWT auth · per-user KB namespacing · jobs  │
                        └───────────────┬──────────────────────────────┘
                                        │
   upload KB ──► DocumentProcessor ──► chunk + embed (all-MiniLM-L6-v2)
                (document-processing.py)        │
                                                ▼
                                        MongoDB (db_operations.py)
                                        users · chunk_embeddings ·
                                        questionnaire · results · metrics
                                                │
   generate ──► QuestionGenerationAgent ────────┘   (questionnaire-generation.py)
                IKB / OOKB / PARTIAL questions, JSON-mode, one call per category
                                                │
   evaluate ──► 3 RAG agents (rag-agent.py) ────┤
                ├─ E2ERAGAgent      (retriever + generator, natural context)
                ├─ RetrieverAgent   (top-k chunks only → recall/precision)
                └─ GeneratorAgent   (forced context: correct / empty / near-miss)
                                                │
                JudgeAgent (evaluator-agent.py) │  LLM classifier → 6 situations
                                                ▼
                MetricsCalculator (deterministic) → E2E-OOKRI + sub-metrics
                InsightsAgent → plain-language diagnosis
```

| Component | File | Role |
|---|---|---|
| Web app + API | `backend/app.py`, `backend/api.py` | SPA host, REST routes, JWT auth, background jobs |
| Doc processing | `backend/document-processing.py` | PDF/PPTX/DOCX → text chunks |
| Retrieval | `backend/api.py` (`_LocalRetriever`), `backend/rag-agent.py` | Hybrid TF-IDF + semantic (`all-MiniLM-L6-v2`), α-weighted |
| Question generation | `backend/questionnaire-generation.py` | LLM authors IKB/OOKB/PARTIAL questions from the KB |
| RAG under test | `backend/rag-agent.py` | Strict & Lenient built-in generators; optional external RAG over HTTP |
| Judge + metrics | `backend/evaluator-agent.py` | LLM-as-judge → deterministic metric aggregation |
| Insights | `backend/insights-agent.py` | Narrates bottleneck (retriever vs generator) |
| LLM runtime | `backend/local_inference.py` | Loads local Qwen2.5 model(s) once, shared by all agents |
| Persistence | `backend/db_operations.py` | MongoDB collections |

**Two built-in RAG postures** are evaluated side-by-side so the score is
interpretable as a *comparison*, not an absolute:

- **Strict** — high-guardrail prompt; refuses on any absent/ambiguous context.
  Expected to be safe (high OOKB abstention) but over-cautious (refuses valid
  IKB questions).
- **Lenient** — low-guardrail prompt; answers even on thin context, drawing on
  parametric knowledge. Expected to be helpful but unsafe (hallucinates on OOKB).

An **external RAG** can also be plugged in via a documented HTTP contract
(`POST /query`, `/retrieve`, `/generate`) for teams who want to test their own
pipeline rather than the built-ins.

---

## 3. Evaluation: methodology and results

### Why this methodology

There is **no labelled ground truth** for "did the system appropriately
abstain" — abstention quality is a judgement about grounding, not a string
match. Three methodologies were on the table:

- **Pure benchmark (exact-match):** rejected. Refusals are paraphrastic ("I
  don't know" vs "the documents don't cover this"); exact-match can't score them,
  and IKB answers are free-text.
- **Human eval:** the gold standard for trust, but not repeatable at the cadence
  a pre-deployment gate needs, and out of scope for a 2-day build.
- **LLM-as-judge + comparison/ablation (chosen):** a rubric-driven judge with a
  fixed **6-class schema** (`ookb_refused`, `ookb_hallucinated`, `ikb_correct`,
  `ikb_refused`, `ikb_hallucinated`, `partial_confabulated`) scores every
  response. The judge emits structured intermediate fields (refusal detected,
  factual alignment, context grounding) *before* the final label to reduce
  classification variance.

The methodology is sound because it pairs the judge with an **ablation** that the
plumbing makes free. Running the **generator with forced-correct context** and
comparing against the **end-to-end** result isolates the failure source: if the
generator behaves well on perfect context but the full system hallucinates,
**retrieval** is the culprit. This "attribution delta" is the most actionable
output of the platform.

### The composite score: E2E-OOKRI

A single deployment-weighted **Out-Of-Knowledge-base Robustness Index**:

```
E2E-OOKRI = α · Safety + (1 − α) · Helpfulness
Safety      = 0.8 · OOKB-AbstentionRate + 0.2 · (1 − IKB-HallucinationRate)
Helpfulness = IKB-Coverage
```

`α` is set by deployment posture (public-facing `0.80`, balanced `0.50`,
internal-use `0.25`), so the same run scores differently for a citizen chatbot
vs an internal research aid. Sub-metrics include retriever Recall@k / Precision@k
and generator G-AR (abstention), G-FAR (over-refusal), G-PCCR (partial-context
confabulation). Full definitions in [`metrics.md`](metrics.md); generation
process in [`question_generation.md`](question_generation.md).

### Sample results (honest)

Run on the **MediShield Life** KB, 40 questions (16 IKB / 24 OOKB incl. PARTIAL),
`α = 0.80` (public-facing), Strict generator. These are **real numbers from a
completed local run** (`knowledge-base/demo_results.json`):

| Metric | Value | Reading |
|---|---:|---|
| E2E OOKB Abstention (E2E-AR) | **41.7%** | Refuses too rarely on out-of-KB questions — unsafe |
| E2E IKB Coverage (E2E-COV) | **62.5%** | Answers most in-KB questions |
| E2E IKB Hallucination (E2E-IK-HR) | **6.3%** | Occasionally wrong on in-KB facts |
| **E2E-OOKRI** | **0.542** | Composite, public-facing weighting |
| Retriever Recall@k / Precision@k | **0.75 / 0.25** | Finds a relevant chunk, but dilutes top-k |
| Generator (forced correct ctx) OOKRI | **0.728** | Far better with perfect context… |
| Generator over-refusal (G-FAR) | **56.3%** | …yet still over-refuses valid questions |
| **Attribution Δ-OOKRI (gen − e2e)** | **+0.186** | **Retrieval is the primary bottleneck** |

**What this tells the operator:** the system is *not safe to ship public-facing*
as configured — it hallucinates on ~58% of OOKB questions — and the biggest lever
is **retrieval quality**, not the generator. Two illustrative cases the judge
caught: it *refused* a premium question whose answer was in context
(over-refusal), and answered a deductible question with `$4,500` when the table
said `$3,500` (read the wrong row). The numbers are **illustrative, not a
leaderboard** — n=40 on a single KB. The Lenient comparison run did **not
complete** in this sample (see Limitations).

---

## 4. Data story

**Source.** A single primary document: the *Information Booklet on MediShield
Life* (`knowledge-base/MediShieldLife.pdf`), administered by the **CPF Board on
behalf of the Ministry of Health (MOH)** under the MediShield Life Scheme Act
2015. Publicly distributed via `medishieldlife.sg`; the committed copy is correct
as at its **1 June 2026** publication date.

**Why this document.** MediShield Life is an ideal RAG stress-test: it is dense
with **precise, easily-confused numbers** — age-banded premiums, ward-class
deductibles ($2,000–$4,500), co-insurance tiers (10/5/3%), subsidy rates, claim
limits — that look interchangeable to a model. A retriever that surfaces the
right *table* but the wrong *row*, or a generator that pattern-matches a nearby
figure, fails in exactly the way OOKB robustness is meant to catch (and did — see
the `$4,500` vs `$3,500` case above).

**How it was processed.** Parsed with `pdfplumber`, chunked, embedded locally
with `all-MiniLM-L6-v2`, stored in MongoDB. The IKB/OOKB/PARTIAL questionnaire is
**synthetically generated by the LLM from the document itself** — every IKB
question records its source chunk as ground truth; OOKB questions are authored to
sit plausibly *just outside* the booklet's scope. Synthetic generation is
defensible here because the questions are grounded in (or deliberately bounded
against) a real authoritative source, not invented from nothing.

**Licensing.** A Singapore Government public information booklet, freely
published for citizen reference. Used here for non-commercial evaluation of a
public-sector tool. No scraping or robots.txt concerns — it is an officially
distributed PDF, used as-is.

**Privacy.** **No personal data.** The booklet contains only public policy
parameters and worked examples with fictional persons ("Mr A", "Insured B"). This
is deliberate: the platform is designed to run over confidential internal KBs, so
the demo dataset is intentionally a fully public document.

**Representativeness / skew.** One document, one domain (national health
insurance), English only. Reported metrics do **not** generalise to other KBs;
they characterise this platform on this booklet. The platform itself is
domain-agnostic — point it at any PDF/PPTX/DOCX corpus.

---

## 5. How to run

The application is **Dockerized** — `docker compose up` spins up everything
needed (the app **and** its MongoDB). The only manual step is creating a `.env`
file, which the compose file loads for the model and database settings.

### Prerequisites (starting from a clean machine)
The compose file bundles MongoDB and downloads the model for you, but the host
machine still needs:

1. **Docker Desktop** (or Docker Engine + Compose v2) **installed and running.**
   - Download: <https://www.docker.com/products/docker-desktop/>.
   - On **Windows**, Docker Desktop requires the **WSL2** backend (the installer
     enables it; a one-time reboot and BIOS virtualization may be needed). Use
     **Linux containers** mode (the default).
   - **Start Docker Desktop and wait until it reports "Engine running" before
     running any `docker` command.** If the engine isn't up you'll see an error
     like `open //./pipe/dockerDesktopLinuxEngine: The system cannot find the
     file specified` — that means Docker Desktop is not started, not a problem
     with this project. Verify with:
     ```bash
     docker version      # must print a "Server:" section, not a pipe error
     ```
2. **Internet access** on first run — Docker pulls the `mongo:7` image and the
   build/boot downloads Python deps and the **Qwen2.5-3B** weights (a public,
   ungated model — no Hugging Face token required).
3. **Disk:** ~15–20 GB free (Python/ML image + model weights + Mongo).
4. **RAM:** the 3B model loads in float32 on CPU (~7–8 GB just for weights). Give
   Docker enough memory — on Windows/macOS raise it under **Docker Desktop →
   Settings → Resources** (≥10 GB recommended) or the container may be killed
   while loading the model.
5. **GPU is optional and not used by default.** The container runs CPU-only,
   which works but is slow (see Limitations). No NVIDIA setup is required to run;
   it's only needed if you later wire in GPU inference for speed.

### Step 1 — Configure `.env`
Copy the template and adjust if needed:

```bash
cp .env.example .env
```

The defaults work out of the box for the Docker path. Key variables:

```bash
# LLM runtime — set BOTH to the SAME model so only one copy loads into RAM
# (the two-model split is not feasible on modest hardware).
LARGE_MODEL=Qwen/Qwen2.5-3B-Instruct
SMALL_MODEL=Qwen/Qwen2.5-3B-Instruct
EMBEDDING_MODEL=all-MiniLM-L6-v2

MONGODB_DB=ookb_benchmark      # connection URI is set automatically by compose
USERNAME=your-username         # owner account for the public /rag/* endpoints
```

### Step 2 — Run with Docker (primary, recommended)
```bash
docker compose up --build
```
This starts two services defined in `docker-compose.yml`:
- **`mongo`** — MongoDB 7, with a persistent `mongo_data` volume.
- **`app`** — the FastAPI app. Its `MONGODB_URI` is pinned to the `mongo` service
  on the compose network, so no manual DB wiring is needed.

On first start the container runs `download_models.py` to fetch the model weights
(several GB — cached in the `hf_cache` volume for subsequent runs), then launches
uvicorn. **First boot is slow** while weights download and the model loads; CPU
inference is the throughput bottleneck (see Limitations). When you see uvicorn
listening, open `http://localhost:8000`.

To stop: `docker compose down` (add `-v` to also wipe the Mongo/KB/model volumes).

### Alternative — Local (without Docker)
For development without containers. A clean machine additionally needs **Python
3.12+** and a **MongoDB** reachable at `MONGODB_URI` (e.g.
`docker run -d -p 27017:27017 mongo:7`, then add
`MONGODB_URI=mongodb://localhost:27017/` to `.env`):

```bash
pip install -r requirements.txt
python download_models.py                       # pre-fetch model weights
py -m uvicorn backend.app:app --reload --port 8000
```

### Step 3 — Use it
1. Open `http://localhost:8000`, **register** an account (use the same name as
   `USERNAME` if you want the public `/rag/*` endpoints to resolve).
2. **Upload** the KB (`knowledge-base/MediShieldLife.pdf` or your own docs).
3. **Generate** a questionnaire (`scale` = total questions, default 40).
4. **Evaluate** — pick `α`, optionally supply external RAG URLs, run.
5. Review the dashboard: composite score, sub-metrics, attribution delta,
   per-question judge reasoning, and the plain-language insights report.

First request is slow while the model loads (it pre-warms in the background on
startup).

---

## 6. Deployment considerations

**Who runs it, and where.** An internal GovTech / agency ML or platform team runs
this as a **pre-deployment quality gate** — a step in CI or a manual sign-off
before a RAG assistant ships, and a periodic regression check after. It is an
internal tool, not a citizen-facing service, so it sits inside the agency
network, close to the KB it evaluates. That locality is the whole point of the
local-model design: confidential knowledge bases never leave the machine.

**Cost / compute footprint.** The expensive resource is LLM inference. A single
40-question evaluation issues on the order of ~200 LLM calls (3 modes × generate +
judge) plus question generation. On CPU this is impractically slow; on one
mid-range GPU (e.g. an L4/A10, ~S$0.50–1.50/hr spot) a full run completes in
minutes and a realistic 200–500 question suite across a few KBs is a sub-dollar
job. Embeddings (`all-MiniLM-L6-v2`) and metrics are negligible. Scaling is
embarrassingly parallel — shard questions across workers. With access to a
larger internally-hosted model, only the judge and question-generator need it;
the RAG-under-test can stay small.

**What I'd monitor live.** Judge-label distribution drift over time, judge
self-consistency on a fixed held-out probe set, question-generation reject/retry
rate, and per-run latency/cost.

**The risk that would keep me up at night.** **A miscalibrated judge silently
passing an unsafe system.** Because the judge is the same class of model as the
generator, a systematic blind spot (e.g. treating a confident hallucination as
"grounded") would inflate safety scores and green-light a system that confabulates
in production — the exact failure the tool exists to prevent. I'd mitigate with a
human-audited calibration set and periodic spot-checks, never treating the score
as ground truth.

---

## 7. Honest limitations

- **Local CPU inference is the bottleneck.** The full pipeline runs end-to-end,
  but on CPU a complete run is impractically slow. The committed Strict-system
  results are from a completed run; the **Lenient comparison run did not finish**
  (its metrics are zeros in `demo_results.json`). A GPU resolves this; the design
  choice was deliberate (data sensitivity) but costs reproducibility on modest
  hardware.
- **One shared model for every LLM role.** Generation, judging, and question
  generation all use the same local model because two differently-sized models
  would not fit in RAM together. Ideally a larger model would handle question
  generation and judging (both need broad KB context) while a smaller model plays
  the RAG-under-test.
- **Judge ≈ generator capability (circularity).** A small local judge scoring a
  small local generator risks shared blind spots. See the deployment risk above.
- **Single KB, small n.** All reported numbers are MediShield Life, n=40. They
  are illustrative of the *method*, not a benchmark of any model.
- **Entrypoint naming.** The real entrypoint is `backend.app:app` (which mounts
  the routes from `backend/api.py`), not `backend.api`. Section 5 gives the
  working commands.
- **Generated-question quality varies** with the local model; the UI allows
  manual editing of questions before evaluation as a backstop.

**What I'd do with no constraints:** run a larger judge/question-gen model on
GPU, add a human-calibrated judge validation set, and expand to multiple KBs and
n≥200 with confidence intervals.

---

## 8. Repository map

```
backend/            FastAPI app, agents, persistence, local inference
frontend/           Vanilla SPA (index.html + static/)
knowledge-base/     MediShield Life PDF, derived chunks, questionnaire, demo results
evaluation-results/ Saved evaluation run artifacts
skills/             kb-question-generator skill (questionnaire generation, packaged)
metrics.md              Full metric definitions
question_generation.md  IKB/OOKB question taxonomy and process
PROCESS.md              How this was built — decisions, pivots, judgment calls
```
