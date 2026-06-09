"""
questionnaire-generation-agent.py

QuestionGenerationAgent — LangGraph ReAct agent that reads kb_chunks.json,
analyses the KB boundary, and generates a mixed IKB/OOKB questionnaire
following the defined taxonomy, writing questionnaire.json.

Model: QUESTION_GENERATION_MODEL (default: mistralai/Mixtral-8x7B-Instruct-v0.1)
Mixtral-8x7B is chosen because structured question generation across nine
distinct categories requires strong instruction following. The MoE architecture
provides GPT-3.5-level quality within the HuggingFace free tier.

Outputs:
    kb_boundary.json   — domain scope, entities, temporal range, topic depth
    questionnaire.json — list of question objects with IKB/OOKB metadata
"""

import json
import os
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

# ── Graph state ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


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

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_TEMPLATE = """You are an expert RAG evaluation question generator.

You will receive knowledge base chunks. Generate a complete IKB/OOKB questionnaire by:
1. Analysing the KB boundary and calling save_boundary.
2. Generating questions for each category (in order below), calling add_questions after each.
3. Calling save_questionnaire once all nine categories are done.

Target question counts:
{distribution}

IKB categories (expected_behavior = "answer") — always record source_chunks:
- I-A Direct recall: answerable from a single chunk, ask about a clear standalone fact.
- I-B Multi-chunk synthesis: requires combining facts from 2+ different chunks.
- I-C Paraphrased retrieval: ask about a KB fact using wording different from the KB.
- I-D Boundary facts: fact is present but obscure, easy for a retriever to miss.

OOKB categories (expected_behavior = "refuse") — must look plausibly answerable:
- A Domain-adjacent: same field as KB, specific topic confirmed absent.
- B Entity substitution: swap a KB entity for a similar entity NOT in the KB.
- C Temporal displacement: shift time reference outside KB's coverage window.
- D Detail escalation: ask for a specific detail on a shallow topic; set source_chunks to near-miss chunk.
- E False presupposition: embed a premise KB never establishes; set source_chunks to near-miss chunk.

OOKB A/B/C: source_chunks = null, expected_answer = null.
OOKB D/E:   source_chunks = near-miss chunk text, expected_answer = null.

Question object schema (for add_questions):
{{
  "question": "...",
  "expected_answer": "string or null",
  "source_documents": ["filename"],
  "source_chunks": ["chunk text"] or null
}}"""


# ── Agent class ───────────────────────────────────────────────────────────────

class QuestionGenerationAgent:
    """
    LangGraph ReAct agent that analyses a KB and generates an IKB/OOKB
    evaluation questionnaire across nine question categories.

    Tools:
        save_boundary      — persists KB domain scope, entities, temporal range
        add_questions      — accumulates generated questions for one category
        save_questionnaire — assigns IDs, adds metadata, writes questionnaire.json

    The agent generates question text using its own language capabilities.
    Tools only handle stateful side-effects so the context window stays lean
    across nine sequential category calls.
    """

    def __init__(self, hf_token: str | None = None, model: str | None = None):
        self.hf_token = hf_token or os.getenv("HF_TOKEN", "")
        self.model = model or os.getenv("QUESTION_GENERATION_MODEL",
                                        "mistralai/Mixtral-8x7B-Instruct-v0.1")
        self._boundary: dict = {}
        self._questions_by_category: dict[str, list] = {}
        self._output_dir: Path = Path(".")

    # ── LLM ──────────────────────────────────────────────────────────────────

    def _build_llm(self) -> ChatHuggingFace:
        endpoint = HuggingFaceEndpoint(
            repo_id=self.model,
            huggingfacehub_api_token=self.hf_token,
            task="text-generation",
            max_new_tokens=4096,
            temperature=0.1,
        )
        return ChatHuggingFace(llm=endpoint)

    # ── Tools (closures over instance state) ─────────────────────────────────

    def _make_tools(self) -> list:

        @tool
        def save_boundary(
            domain_scope: list[str],
            entities: list[str],
            temporal_range: dict,
            deep_topics: list[str],
            shallow_topics: list[str],
        ) -> str:
            """
            Save KB boundary analysis. Call this before generating any questions.
            domain_scope: main subject areas covered.
            entities: named entities (orgs, policies, laws, products, people).
            temporal_range: {start, end, notes} describing the KB time window.
            deep_topics: topics with many chunks of detail.
            shallow_topics: topics mentioned in only 1-2 chunks.
            """
            self._boundary = {
                "domain_scope": domain_scope, "entities": entities,
                "temporal_range": temporal_range, "deep_topics": deep_topics,
                "shallow_topics": shallow_topics,
            }
            path = self._output_dir / "kb_boundary.json"
            path.write_text(json.dumps(self._boundary, indent=2, ensure_ascii=False),
                            encoding="utf-8")
            return f"Boundary saved to {path}."

        @tool
        def add_questions(category: str, questions: list[dict]) -> str:
            """
            Accumulate generated questions for one category.
            category: one of I-A, I-B, I-C, I-D, A, B, C, D, E.
            questions: list of question objects (question, expected_answer,
                       source_documents, source_chunks).
            Call once per category in order: I-A, I-B, I-C, I-D, A, B, C, D, E.
            """
            self._questions_by_category[category] = questions
            total = sum(len(v) for v in self._questions_by_category.values())
            return f"Added {len(questions)} questions for {category}. Total: {total}."

        @tool
        def save_questionnaire(output_path: str) -> str:
            """
            Assign IDs, add metadata, and write questionnaire.json.
            Call this once all nine categories have been added.
            """
            all_questions = []
            q_id = 1
            for category in BASE_DISTRIBUTION:
                meta = CATEGORY_META[category]
                for q in self._questions_by_category.get(category, []):
                    all_questions.append({
                        "id": f"q_{q_id:03d}",
                        "question": q.get("question", ""),
                        "type": meta["type"],
                        "category": category,
                        "difficulty": meta["difficulty"],
                        "expected_behavior": "answer" if meta["type"] == "IKB" else "refuse",
                        "expected_answer": q.get("expected_answer"),
                        "source_documents": q.get("source_documents", []),
                        "source_chunks": q.get("source_chunks"),
                        "rag_query": {"query": q.get("question", "")},
                    })
                    q_id += 1

            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(all_questions, indent=2, ensure_ascii=False),
                           encoding="utf-8")
            ikb = sum(1 for q in all_questions if q["type"] == "IKB")
            ookb = sum(1 for q in all_questions if q["type"] == "OOKB")
            return json.dumps({"status": "saved", "total": len(all_questions),
                               "ikb": ikb, "ookb": ookb, "path": str(out)})

        return [save_boundary, add_questions, save_questionnaire]

    # ── Graph ─────────────────────────────────────────────────────────────────

    def _build_graph(self, tools: list):
        llm_with_tools = self._build_llm().bind_tools(tools)
        tool_node = ToolNode(tools)

        def agent_node(state: AgentState) -> dict:
            return {"messages": [llm_with_tools.invoke(state["messages"])]}

        def should_continue(state: AgentState) -> str:
            return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END

        graph = StateGraph(AgentState)
        graph.add_node("agent", agent_node)
        graph.add_node("tools", tool_node)
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        graph.add_edge("tools", "agent")
        return graph.compile()

    # ── Distribution helper ───────────────────────────────────────────────────

    @staticmethod
    def _scale_distribution(total: int) -> dict[str, int]:
        factor = total / 100
        counts = {cat: max(1, round(n * factor)) for cat, n in BASE_DISTRIBUTION.items()}
        diff = total - sum(counts.values())
        if diff != 0:
            counts[max(counts, key=counts.get)] += diff
        return counts

    # ── Public interface ──────────────────────────────────────────────────────

    def run(
        self,
        chunks_path: str | Path,
        output_path: str | Path | None = None,
        scale: int = 100,
    ) -> list[dict]:
        """
        Generate a questionnaire from kb_chunks.json.

        Args:
            chunks_path: path to kb_chunks.json from DocumentProcessingAgent
            output_path: destination for questionnaire.json
                         (defaults to same directory as chunks_path)
            scale:       target total question count (default 100, min 20)

        Returns:
            list of question dicts written to questionnaire.json
        """
        chunks_path = Path(chunks_path)
        output_path = Path(output_path) if output_path else chunks_path.parent / "questionnaire.json"
        scale = max(20, scale)

        self._boundary = {}
        self._questions_by_category = {}
        self._output_dir = output_path.parent

        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        if not chunks:
            raise ValueError(f"No chunks found in {chunks_path}")

        distribution = self._scale_distribution(scale)
        dist_lines = "\n".join(f"  - {cat}: {n}" for cat, n in distribution.items())
        system_prompt = _SYSTEM_TEMPLATE.format(distribution=dist_lines)

        chunks_context = "\n\n---\n\n".join(
            f"[{c['source']} | chunk {c['chunk_id']}]\n{c['text']}" for c in chunks
        )

        tools = self._make_tools()
        graph = self._build_graph(tools)

        graph.invoke(
            {"messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=(
                    f"Knowledge base chunks ({len(chunks)} total):\n\n{chunks_context}\n\n"
                    f"Output path: {output_path}\n\n"
                    "Begin: analyse the boundary, then generate all nine question categories."
                )),
            ]},
            {"recursion_limit": 60},
        )

        if output_path.exists():
            return json.loads(output_path.read_text(encoding="utf-8"))

        # Fallback: assemble from accumulated state
        all_questions = []
        q_id = 1
        for category in BASE_DISTRIBUTION:
            meta = CATEGORY_META[category]
            for q in self._questions_by_category.get(category, []):
                all_questions.append({
                    "id": f"q_{q_id:03d}",
                    "question": q.get("question", ""),
                    "type": meta["type"],
                    "category": category,
                    "difficulty": meta["difficulty"],
                    "expected_behavior": "answer" if meta["type"] == "IKB" else "refuse",
                    "expected_answer": q.get("expected_answer"),
                    "source_documents": q.get("source_documents", []),
                    "source_chunks": q.get("source_chunks"),
                    "rag_query": {"query": q.get("question", "")},
                })
                q_id += 1
        output_path.write_text(json.dumps(all_questions, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        return all_questions
