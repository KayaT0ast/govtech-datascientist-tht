"""
insights-agent.py

InsightsAgent — generates a structured LLM diagnostic report from benchmark
metrics, classifying failures into two root causes:

    Retrieval errors   — the retriever failed to surface the right chunks,
                         causing downstream hallucination or over-refusal
    Generator errors   — the generator hallucinated or over-refused despite
                         having correct or sufficient context

The agent uses the attribution delta metrics (delta_hallucination,
delta_refusal from evaluator-agent.py) to determine whether failures are
retrieval-side or generation-side, then generates actionable recommendations.

Model: QUESTION_GENERATION_MODEL (default: mistralai/Mixtral-8x7B-Instruct-v0.1)
Mixtral-8x7B is reused here because analytical synthesis of structured metrics
is an instruction-following task well within its capability, avoiding the need
for the heavier Qwen judge model.
"""

import os
from dataclasses import dataclass

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

from dotenv import load_dotenv
from local_inference import LocalChatClient

load_dotenv()

# ── Diagnosis thresholds ──────────────────────────────────────────────────────

_RETRIEVAL_BLAME_THRESHOLD = 0.05   # delta_hallucination > this → retriever is primary cause
_GENERATOR_BLAME_THRESHOLD = 0.10   # G-FAR > this → generator is over-refusing
_HALLUCINATION_SEVERE = 0.30        # E2E-HR > this → severe hallucination
_REFUSAL_SEVERE = 0.25              # G-FAR > this → severe over-refusal
_PCCR_SEVERE = 0.30                 # G-PCCR > this → severe confabulation on partial context


# ── Structured output ─────────────────────────────────────────────────────────

@dataclass
class InsightsReport:
    primary_bottleneck: str          # "retrieval" | "generation" | "both" | "none"
    safety_status: str               # "safe" | "at_risk" | "unsafe"
    helpfulness_status: str          # "helpful" | "over-refusing" | "unreliable"
    retrieval_diagnosis: str         # plain text
    generation_diagnosis: str        # plain text
    recommendations: list[str]       # 3 actionable items
    full_narrative: str              # complete LLM-generated diagnostic


# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a senior RAG Solutions Architect writing a professional benchmark diagnostic. "
    "Analyse the provided metrics and write a concise (200-300 word) structured report. "
    "Your report must explicitly address:\n"
    "1. Safety: OOKB refusal rate and hallucination risk for public deployment.\n"
    "2. Helpfulness: IKB coverage and whether the system is over-refusing.\n"
    "3. Attribution: use delta_hallucination and delta_refusal to determine if failures "
    "are caused by the retriever (can't find right chunks) or the generator "
    "(hallucinating or refusing despite correct context).\n"
    "4. Recommendations: exactly 3 specific engineering actions, ordered by priority.\n"
    "Be direct and professional. No padding."
)


def _build_user_prompt(metrics: dict, alpha: float) -> str:
    e2e = metrics.get("e2e", {})
    gen = metrics.get("generator", {})
    ret = metrics.get("retriever", {})
    attr = metrics.get("attribution", {})

    deployment = (
        "public-facing" if alpha >= 0.7
        else "internal-use" if alpha <= 0.3
        else "balanced"
    )

    return (
        f"Deployment type: {deployment} (α = {alpha})\n\n"
        f"E2E Metrics:\n"
        f"  OOKB Abstention Rate (E2E-AR):    {e2e.get('E2E_AR', 0):.1%}\n"
        f"  OOKB Hallucination Rate (E2E-HR): {e2e.get('E2E_HR', 0):.1%}\n"
        f"  IKB Coverage (E2E-COV):           {e2e.get('E2E_COV', 0):.1%}\n"
        f"  IKB Hallucination Rate:           {e2e.get('E2E_IK_HR', 0):.1%}\n"
        f"  IKB Refusal Rate (safe misses):   {e2e.get('E2E_IK_RR', 0):.1%}\n"
        f"  Composite OOKRI:                  {e2e.get('E2E_OOKRI', 0):.3f}\n\n"
        f"Retrieval Metrics:\n"
        f"  Recall@k:    {ret.get('R_Rec_k', 0):.1%}\n"
        f"  Precision@k: {ret.get('R_Prec_k', 0):.1%}\n\n"
        f"Generator Metrics:\n"
        f"  G-AR   (abstention on empty context):   {gen.get('G_AR', 0):.1%}\n"
        f"  G-FAR  (false abstention on good ctx):  {gen.get('G_FAR', 0):.1%}\n"
        f"  G-PCCR (confabulation on near-miss):    {gen.get('G_PCCR', 0):.1%}\n\n"
        f"Attribution Delta (E2E − Generator):\n"
        f"  Δ Hallucination: {attr.get('delta_hallucination', 0):+.1%} "
        f"(positive = retriever causing hallucinations)\n"
        f"  Δ Refusal:       {attr.get('delta_refusal', 0):+.1%} "
        f"(positive = retriever causing over-refusal)\n"
    )


# ── Agent class ───────────────────────────────────────────────────────────────

class InsightsAgent:
    """
    Generates a structured diagnostic report from benchmark metrics.

    Performs two passes:
        1. Rule-based pre-diagnosis — classifies bottleneck type and severity
           from metric thresholds without an LLM call (fast, always available).
        2. LLM narrative — generates a full professional diagnostic using the
           pre-diagnosis as grounding context.

    The pre-diagnosis ensures the LLM report is anchored to the actual numbers
    rather than producing generic advice.
    """

    def __init__(self, hf_token: str | None = None, model: str | None = None):
        self.client = LocalChatClient(size="small")

    # ── Rule-based pre-diagnosis ──────────────────────────────────────────────

    @staticmethod
    def _pre_diagnose(metrics: dict) -> dict:
        """Classify bottleneck and severity from metric thresholds."""
        e2e = metrics.get("e2e", {})
        gen = metrics.get("generator", {})
        attr = metrics.get("attribution", {})

        delta_hall = attr.get("delta_hallucination", 0)
        delta_ref = attr.get("delta_refusal", 0)
        g_far = gen.get("G_FAR", 0)
        g_pccr = gen.get("G_PCCR", 0)
        e2e_hr = e2e.get("E2E_HR", 0)
        e2e_ar = e2e.get("E2E_AR", 0)

        retrieval_at_fault = (delta_hall > _RETRIEVAL_BLAME_THRESHOLD or
                              delta_ref > _RETRIEVAL_BLAME_THRESHOLD)
        generation_at_fault = (g_far > _GENERATOR_BLAME_THRESHOLD or
                               g_pccr > _PCCR_SEVERE)

        if retrieval_at_fault and generation_at_fault:
            bottleneck = "both"
        elif retrieval_at_fault:
            bottleneck = "retrieval"
        elif generation_at_fault:
            bottleneck = "generation"
        else:
            bottleneck = "none"

        safety = ("unsafe" if e2e_hr > _HALLUCINATION_SEVERE
                  else "at_risk" if e2e_hr > 0.10
                  else "safe")

        helpfulness = ("over-refusing" if g_far > _REFUSAL_SEVERE
                       else "unreliable" if e2e.get("E2E_IK_HR", 0) > _HALLUCINATION_SEVERE
                       else "helpful")

        retrieval_diag = []
        if delta_hall > _RETRIEVAL_BLAME_THRESHOLD:
            retrieval_diag.append(
                f"Retriever is causing hallucinations (Δhall = {delta_hall:+.1%}). "
                "The generator hallucinates when given poor retrieval but behaves "
                "correctly with forced correct context — fix retrieval first."
            )
        if delta_ref > _RETRIEVAL_BLAME_THRESHOLD:
            retrieval_diag.append(
                f"Retriever is causing over-refusal (Δref = {delta_ref:+.1%}). "
                "The generator refuses when it receives poor context but answers "
                "correctly with forced context — retrieval recall is insufficient."
            )
        if not retrieval_diag:
            retrieval_diag.append("Retrieval quality is not the primary failure mode.")

        generation_diag = []
        if g_far > _GENERATOR_BLAME_THRESHOLD:
            generation_diag.append(
                f"Generator over-refuses on good context (G-FAR = {g_far:.1%}). "
                "The system prompt is likely too conservative — consider softening "
                "the refusal threshold."
            )
        if g_pccr > _PCCR_SEVERE:
            generation_diag.append(
                f"Generator confabulates on partial context (G-PCCR = {g_pccr:.1%}). "
                "The generator fills in missing retrieval gaps with invented facts — "
                "strengthen the 'hedge when uncertain' instruction."
            )
        if e2e_ar < 0.70:
            generation_diag.append(
                f"OOKB abstention rate is low (E2E-AR = {e2e_ar:.1%}). "
                "The system is not refusing enough on out-of-KB questions."
            )
        if not generation_diag:
            generation_diag.append("Generator behaviour is within acceptable bounds.")

        return {
            "primary_bottleneck": bottleneck,
            "safety_status": safety,
            "helpfulness_status": helpfulness,
            "retrieval_diagnosis": " ".join(retrieval_diag),
            "generation_diagnosis": " ".join(generation_diag),
        }

    # ── LLM narrative ─────────────────────────────────────────────────────────

    def _generate_narrative(self, metrics: dict, pre_diag: dict, alpha: float) -> tuple[str, list[str]]:
        """Generate full narrative and extract 3 recommendations."""
        grounding = (
            f"Pre-diagnosis summary:\n"
            f"  Primary bottleneck: {pre_diag['primary_bottleneck']}\n"
            f"  Safety status:      {pre_diag['safety_status']}\n"
            f"  Helpfulness:        {pre_diag['helpfulness_status']}\n"
            f"  Retrieval issue:    {pre_diag['retrieval_diagnosis']}\n"
            f"  Generation issue:   {pre_diag['generation_diagnosis']}\n\n"
        )
        user_prompt = grounding + _build_user_prompt(metrics, alpha)

        try:
            resp = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=600,
                temperature=0.3,
            )
            narrative = resp.choices[0].message.content.strip()
        except Exception as e:
            narrative = f"Narrative generation failed: {e}"

        # Extract numbered recommendations from narrative
        recs = []
        for line in narrative.split("\n"):
            stripped = line.strip()
            if stripped and stripped[0].isdigit() and stripped[1:3] in (". ", ") "):
                recs.append(stripped[3:].strip())
            if len(recs) == 3:
                break

        if len(recs) < 3:
            recs = [
                "Improve retrieval recall by tuning the embedding model or chunk size.",
                "Adjust the generator's refusal threshold via system prompt tuning.",
                "Re-evaluate with a larger question set to increase metric confidence.",
            ][:3]

        return narrative, recs

    # ── Public interface ──────────────────────────────────────────────────────

    def generate(self, metrics: dict, alpha: float | None = None) -> InsightsReport:
        """
        Generate a full diagnostic report from MetricsCalculator output.

        Args:
            metrics: output of MetricsCalculator.compute()
            alpha:   deployment weight used in evaluation (reads from metrics if None)

        Returns:
            InsightsReport dataclass
        """
        alpha = alpha if alpha is not None else metrics.get("e2e", {}).get("alpha", 0.50)

        pre_diag = self._pre_diagnose(metrics)
        narrative, recommendations = self._generate_narrative(metrics, pre_diag, alpha)

        return InsightsReport(
            primary_bottleneck=pre_diag["primary_bottleneck"],
            safety_status=pre_diag["safety_status"],
            helpfulness_status=pre_diag["helpfulness_status"],
            retrieval_diagnosis=pre_diag["retrieval_diagnosis"],
            generation_diagnosis=pre_diag["generation_diagnosis"],
            recommendations=recommendations,
            full_narrative=narrative,
        )

    def generate_dict(self, metrics: dict, alpha: float | None = None) -> dict:
        """Same as generate() but returns a plain dict (JSON-serialisable)."""
        report = self.generate(metrics, alpha)
        return {
            "primary_bottleneck": report.primary_bottleneck,
            "safety_status": report.safety_status,
            "helpfulness_status": report.helpfulness_status,
            "retrieval_diagnosis": report.retrieval_diagnosis,
            "generation_diagnosis": report.generation_diagnosis,
            "recommendations": report.recommendations,
            "full_narrative": report.full_narrative,
        }
