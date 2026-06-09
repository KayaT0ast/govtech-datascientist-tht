# PROCESS — How I built the OOKB Robustness Benchmark

A narrative of the decisions, pivots, and judgment calls behind this project —
including the ones that didn't work out. Where the repo and this document
disagree with an earlier draft, this document reflects what the code actually
does today.

---

## Choosing the problem

I started from a conviction about *which* RAG failure matters most in the public
sector. Most RAG tooling measures whether a system answers correctly. The failure
that actually destroys trust is the opposite: a confident, fluent answer to a
question the knowledge base can't support. For a citizen-facing or internal
policy assistant, a fabricated number is worse than an honest "I don't know."

So I scoped to **Out-of-Knowledge-Base (OOKB) robustness** (Annex A #15):
measuring *abstention discipline*, not just answer quality. This also fit the
time budget — it's a sharp, finishable slice rather than "build a RAG product."

**Domain.** I chose the **MediShield Life** information booklet because it has
exactly the properties an OOKB stress test needs: it is a single authoritative
public document, contains **no personal data**, and is saturated with
interchangeable-looking numbers (age-banded premiums, ward-class deductibles,
co-insurance tiers) that a RAG system can easily confuse. That makes it a genuine
stress test — and in practice the system did get caught reading the wrong row of
a deductible table.

---

## The model-backend journey (the biggest pivot)

This is where I spent the most judgment.

**Attempt 1 — OpenRouter, hosted.** I first wired question generation and
evaluation to OpenRouter's `meta-llama/llama-3.3-70b-instruct:free`. It worked,
but two things pushed me off it:

1. **Free-tier request limits.** A single evaluation fans out into hundreds of
   LLM calls (three modes × generate + judge, plus question generation). The free
   model's rate limits made full runs stall.
2. **Data sensitivity.** The whole point of this tool is to evaluate RAG systems
   built over *internal* knowledge bases — which in an agency context may be
   confidential. A benchmark that ships those documents to a hosted API
   undermines its own use case. That's a task-fit argument, not a "local is
   cool" one.

**Attempt 2 — two local models.** I switched to local inference via Hugging Face
`transformers`. My intended design used **two** models: a larger one for
question generation and judging (both need broad context over the KB to decide
what's in- vs out-of-scope and to author plausible near-miss questions), and a
smaller one to play the RAG-under-test and write insights.

**What I dropped.** I couldn't hold both models in RAM at once on my hardware.
Rather than ship something that only runs on a big machine, I collapsed to **one
shared model for every LLM role** (`backend/local_inference.py` loads a single
pipeline; setting `LARGE_MODEL == SMALL_MODEL` loads one copy). It's a real
compromise — a small model judging a small generator risks shared blind spots —
and I called it out as a limitation rather than papering over it. **Given GovTech
compute and access to a larger internally-hosted model, I'd put the big model
back on question generation and judging only.**

The honest current state: the pipeline runs end-to-end, but on CPU it's slow
enough that the committed sample has a complete **Strict** run and an
**incomplete Lenient** one (zeros). I left that visible.

---

## Question generation — a second dropped approach

My first implementation had the LLM build the questionnaire through **multi-turn
tool-calling** (an `add_questions` tool it would call repeatedly). It was
unreliable: models skipped tool calls, hit recursion limits, or emitted truncated
tool-call JSON. I dropped it for **direct per-category JSON-mode calls** — one
focused call per question type (IKB / OOKB / PARTIAL), each asked to emit a plain
JSON array, with up to two retries on parse failure. Less elegant, far more
reliable. This is documented at the top of
`backend/questionnaire-generation.py`.

---

## Architecture judgment calls

- **FastAPI + vanilla HTML/CSS/JS over React.** I wanted a single container that
  Dockerizes cleanly and a grader can run without a Node toolchain. The frontend
  is a hand-written SPA; the cost is no component framework, which is fine at this
  scope.
- **In-memory hybrid retriever over a cloud vector DB.** TF-IDF + `all-MiniLM-L6-v2`
  semantic scores, α-weighted. Rejecting Pinecone et al. kept the system
  self-contained and consistent with the data-sensitivity stance.
- **MongoDB for persistence and multi-user namespacing.** Every collection keys on
  `username` + `rag_system_name`, so one instance can host several users and KBs.
- **Deterministic metrics, separated from the judge.** The LLM only produces
  per-response *labels*; all scoring math (`MetricsCalculator`) is pure and
  reproducible. I split responsibilities so that the non-deterministic part is as
  small and auditable as possible.
- **Strict vs Lenient comparison + attribution delta.** I deliberately evaluate
  two generator postures and also run the generator on *forced-correct* context.
  Comparing end-to-end against forced-context isolates whether the retriever or
  the generator is the weak link — the single most actionable output, and the
  thing that turns an LLM-judge score into a diagnosis.

---

## Tooling

- **Coding agent (Claude Code)** for scaffolding the FastAPI routes, agent
  classes, and the SPA, and for iterating on prompts — with manual review of the
  output against a quality bar.
- **`pdfplumber`** for PDF parsing, **`sentence-transformers`** for embeddings,
  **`transformers` + `accelerate`** for local LLM inference, **`pymongo`** for
  persistence.
- Judgment about *what to build and whether it actually works* stayed with me;
  the agent accelerated the writing, not the decisions.

---

## Where I exercised judgment, in one line each

- Scoped to abstention discipline because it's the under-measured,
  trust-critical failure mode — not the easiest thing to demo.
- Chose local inference on a defensible ground (data sensitivity + free-tier
  limits), then accepted a real quality cost (one shared model) rather than
  pretend the two-model design shipped.
- Designed the judge to emit structured reasoning before its label to cut
  variance, while staying aware it can't validate itself.
- Reported the retrieval-bottleneck finding and the incomplete Lenient run
  honestly, because an evaluation tool that isn't honest about its own results
  has no standing to judge anyone else's.
