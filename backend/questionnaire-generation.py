"""
questionnaire-generation.py

QuestionGenerationAgent — generates a mixed IKB/OOKB questionnaire from KB
chunks using direct per-category JSON-mode LLM calls.

Each of the 9 question categories is generated in a separate focused LLM
call (JSON array output).  This avoids the multi-turn tool-calling pattern
that was unreliable: models would skip add_questions calls, hit recursion
limits, or produce truncated tool-call JSON.

Per-call workflow:
  1. Build system prompt: "output only a JSON array"
  2. Build user prompt: category description + schema + KB chunks
  3. Parse JSON array from response (strips markdown fences if present)
  4. Retry up to 2 times on parse failure
  5. Resolve source_chunk_ids → source_chunks text locally

Outputs:
    kb_chunks.json     — already written by DocumentProcessor
    questionnaire.json — list of question objects with IKB/OOKB metadata
"""

import json
import os
import random
import re
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path

from dotenv import load_dotenv
from local_inference import LocalChatClient

load_dotenv()


# ── Question taxonomy ─────────────────────────────────────────────────────────

BASE_DISTRIBUTION = {
    "IKB":     40,
    "OOKB":    40,
    "PARTIAL": 20,
}

CATEGORY_META = {
    "IKB":     {"type": "IKB",  "difficulty": "mixed"},
    "OOKB":    {"type": "OOKB", "difficulty": "mixed"},
    "PARTIAL": {"type": "OOKB", "difficulty": "hard"},
}

_CATEGORY_DESCRIPTIONS = {
    "IKB": (
        "Generate questions that are ANSWERABLE from the KB. "
        "Mix easy single-chunk questions with harder multi-chunk or paraphrased ones. "
        "Set source_chunk_ids to the relevant chunk(s). "
        "Set expected_answer to a concise factual string."
    ),
    "OOKB": (
        "Generate questions that are NOT ANSWERABLE from the KB — "
        "the topic is absent, the entity is not mentioned, or the time period "
        "is outside the KB's coverage. Questions should look plausibly answerable. "
        "Set source_chunk_ids to null. Set expected_answer to null."
    ),
    "PARTIAL": (
        "Generate questions where the KB covers the general topic but NOT the "
        "specific detail asked, or the question embeds a false premise the KB "
        "never establishes. Set source_chunk_ids to the ID of the nearest "
        "near-miss KB chunk (closest topic but doesn't actually answer). "
        "Set expected_answer to null."
    ),
}

_SYSTEM_PROMPT = (
    "You are an expert RAG evaluation question generator. "
    "Output ONLY a valid JSON array with no markdown fences, "
    "no commentary, and no text outside the array."
)


# ── Agent ─────────────────────────────────────────────────────────────────────

class QuestionGenerationAgent:
    """
    Generates an IKB/OOKB evaluation questionnaire from KB chunks.

    Uses 9 direct JSON-mode LLM calls (one per category) instead of
    multi-turn tool-calling, which is simpler and far more reliable.

    The public interface is identical to the previous LangGraph version:
        agent = QuestionGenerationAgent()
        questions = agent.run(chunks_path, output_path, scale=40)
    """

    def __init__(self, hf_token: str | None = None, model: str | None = None):
        self._client = LocalChatClient(size="large")

    # ── Distribution ──────────────────────────────────────────────────────────

    @staticmethod
    def _scale_distribution(total: int) -> dict[str, int]:
        factor = total / 100
        counts = {cat: max(1, round(n * factor)) for cat, n in BASE_DISTRIBUTION.items()}
        diff = total - sum(counts.values())
        if diff != 0:
            counts[max(counts, key=counts.get)] += diff
        return counts

    # ── Per-category generation ───────────────────────────────────────────────

    def _generate_category(
        self,
        category: str,
        count: int,
        meta: dict,
        chunks_context: str,
        retries: int = 2,
    ) -> list[dict]:
        """
        Generate `count` questions for one category via a single LLM call.
        Returns a list of raw question dicts (no IDs, no source_chunks text yet).
        """
        is_ikb = meta["type"] == "IKB"
        desc = _CATEGORY_DESCRIPTIONS[category]

        if is_ikb:
            source_note = (
                "For EVERY question set source_chunk_ids to the chunk ID(s) "
                "whose text contains the answer. "
                "Derive IDs from the chunk headers: "
                "[MediShieldLife.pdf | chunk 5] → \"MediShieldLife.pdf_chunk_5\". "
                "Set expected_answer to the concise factual answer."
            )
        elif category == "PARTIAL":
            source_note = (
                "Set source_chunk_ids to the ID of the nearest KB chunk "
                "(the best near-miss — closest topic but doesn't actually answer). "
                "Set expected_answer to null."
            )
        else:
            source_note = "Set source_chunk_ids to null. Set expected_answer to null."

        schema = (
            '{"question": "...", '
            '"expected_answer": "concise answer string or null", '
            '"source_documents": ["filename.pdf"], '
            '"source_chunk_ids": ["filename.pdf_chunk_N"] or null}'
        )

        user_msg = (
            f"Generate exactly {count} questions for category {category}.\n\n"
            f"Category description: {desc}\n\n"
            f"Rules:\n"
            f"  - {source_note}\n"
            f"  - Every question must look plausibly answerable from the KB.\n"
            f"  - Do not repeat questions.\n\n"
            f"Output: a JSON array of exactly {count} objects matching this schema:\n"
            f"[{schema}, ...]\n\n"
            f"Knowledge base chunks:\n{chunks_context}"
        )

        for attempt in range(retries + 1):
            try:
                max_tok = max(256, count * 100 + 150)
                print(f"    [{category}] attempt {attempt + 1}/{retries + 1} — calling LLM (max_tokens={max_tok}) …")
                resp = self._client.chat_completion(
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=max_tok,
                    temperature=0.3,
                )
                text = (resp.choices[0].message.content or "").strip()
                print(f"    [{category}] LLM returned {len(text)} chars: {text[:120]!r} …")
                # Strip markdown fences if the model added them
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
                # Extract the outermost JSON array
                s, e = text.find("["), text.rfind("]")
                if s != -1 and e != -1:
                    parsed = json.loads(text[s : e + 1])
                    if isinstance(parsed, list) and len(parsed) > 0:
                        print(f"    [{category}] parsed {len(parsed)} questions OK.")
                        return parsed
                    print(f"    [{category}] Warning: empty array, retrying …")
                else:
                    print(f"    [{category}] Warning: no JSON array found in response, retrying …")
            except Exception as exc:
                if attempt < retries:
                    print(f"    [{category}] Retry {attempt + 1}/{retries}: {exc}")
                else:
                    print(f"    [{category}] Failed after all retries: {exc}")
        return []

    # ── Public interface ──────────────────────────────────────────────────────

    def run(
        self,
        chunks_path: str | Path,
        output_path: str | Path | None = None,
        scale: int = 100,
        progress_callback=None,   # callable(message: str, progress: int) or None
    ) -> list[dict]:
        """
        Generate a questionnaire from kb_chunks.json.

        Args:
            chunks_path: path to kb_chunks.json produced by DocumentProcessor
            output_path: destination for questionnaire.json
                         (defaults to same directory as chunks_path)
            scale:       target total question count (default 100, min 20)

        Returns:
            list of question dicts written to questionnaire.json
        """
        chunks_path = Path(chunks_path)
        output_path = (
            Path(output_path) if output_path
            else chunks_path.parent / "questionnaire.json"
        )
        scale = max(20, scale)

        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        if not chunks:
            raise ValueError(f"No chunks found in {chunks_path}")

        # Build ID → text lookup so we can resolve source_chunk_ids locally
        chunk_lookup: dict[str, str] = {
            f"{c['source']}_chunk_{c['chunk_id']}": c["text"]
            for c in chunks
        }
        # Cap context to avoid overflowing the model's context window.
        # Passing all chunks at once easily exceeds 8k tokens and causes silent failures.
        _sample = random.sample(chunks, min(25, len(chunks)))
        chunks_context = "\n\n---\n\n".join(
            f"[{c['source']} | chunk {c['chunk_id']}]\n{c['text']}" for c in _sample
        )

        distribution = self._scale_distribution(scale)
        all_questions: list[dict] = []
        q_id = 1
        total_categories = len(distribution)
        done_categories = 0

        for category, count in distribution.items():
            meta = CATEGORY_META[category]
            print(f"  Generating {count} × {category} ({meta['type']}) …")
            if progress_callback:
                pct = 50 + int(40 * done_categories / total_categories)
                progress_callback(
                    f"Generating questionnaire — category {category} "
                    f"({done_categories}/{total_categories}, {len(all_questions)} questions so far) …",
                    pct,
                )
            raw = self._generate_category(category, count, meta, chunks_context)
            done_categories += 1

            for q in raw:
                ids = q.get("source_chunk_ids") or []
                if isinstance(ids, str):
                    ids = [ids]  # handle LLM returning a string instead of list
                texts = [chunk_lookup[cid] for cid in ids if cid in chunk_lookup]

                all_questions.append({
                    "id": f"q_{q_id:03d}",
                    "question": q.get("question", ""),
                    "type": meta["type"],
                    "category": category,
                    "difficulty": meta["difficulty"],
                    "expected_behavior": "answer" if meta["type"] == "IKB" else "refuse",
                    "expected_answer": q.get("expected_answer"),
                    "source_documents": q.get("source_documents", []),
                    "source_chunks": texts or None,
                    "source_chunk_ids": ids or None,
                    "rag_query": {"query": q.get("question", "")},
                })
                q_id += 1

            print(f"    → {len(raw)} added  (total so far: {len(all_questions)}) ✓")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(all_questions, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        ikb_n  = sum(1 for q in all_questions if q["type"] == "IKB")
        ookb_n = sum(1 for q in all_questions if q["type"] == "OOKB")
        print(
            f"  Questionnaire saved: {len(all_questions)} questions "
            f"({ikb_n} IKB, {ookb_n} OOKB) → {output_path}"
        )
        return all_questions
