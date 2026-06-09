"""
local_inference.py

Two models, both loaded once at startup, shared by every agent.

Configure via .env:
    LARGE_MODEL=Qwen/Qwen2.5-7B-Instruct   (judge, question generation)
    SMALL_MODEL=Qwen/Qwen2.5-3B-Instruct   (RAG generator, insights)

Both pipelines are initialised when this module is first imported.
Agents that ask for the large model get _LARGE_PIPELINE;
agents that ask for the small model get _SMALL_PIPELINE.
No model is ever loaded or downloaded more than once per process.

Hardware notes:
    GPU (CUDA)  — float16, device_map="auto".
    CPU only    — float32, slow.  Both models are kept in RAM simultaneously,
                  so ~20 GB RAM is needed for the 7B + 3B pair.
                  If memory is tight, set LARGE_MODEL=Qwen/Qwen2.5-3B-Instruct
                  so only one model is downloaded.
"""

import os
import threading
from dataclasses import dataclass, field

import torch
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()

# ── Model config (read at import time, loaded on first use) ───────────────────

_LARGE_MODEL_ID = os.getenv("LARGE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
_SMALL_MODEL_ID = os.getenv("SMALL_MODEL", "Qwen/Qwen2.5-3B-Instruct")

_CUDA    = torch.cuda.is_available()
_DTYPE   = torch.float16 if _CUDA else torch.float32
_DEV_MAP = "auto" if _CUDA else "cpu"

# Pipelines are None until first use; _lock prevents concurrent loading
_LARGE_PIPELINE  = None
_SMALL_PIPELINE  = None
_loaded          = False
_load_error: str = ""          # set on failure; prevents silent retries
_lock            = threading.Lock()


def _load_pipeline(model_id: str):
    print(f"\n{'='*60}")
    print(f"[local_inference] Loading '{model_id}' …")
    print(f"  dtype={_DTYPE}  device_map={_DEV_MAP!r}")
    print(f"{'='*60}")
    p = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=_DTYPE,
        device_map=_DEV_MAP,
    )
    print(f"[local_inference] '{model_id}' loaded and ready.\n")
    return p


def _ensure_loaded():
    """Load both pipelines on first call; raises RuntimeError on failure."""
    global _LARGE_PIPELINE, _SMALL_PIPELINE, _loaded, _load_error
    if _loaded:
        return
    if _load_error:
        raise RuntimeError(f"Model loading previously failed: {_load_error}")
    with _lock:
        if _loaded:
            return
        if _load_error:
            raise RuntimeError(f"Model loading previously failed: {_load_error}")
        try:
            _LARGE_PIPELINE = _load_pipeline(_LARGE_MODEL_ID)
            if _SMALL_MODEL_ID == _LARGE_MODEL_ID:
                _SMALL_PIPELINE = _LARGE_PIPELINE
            else:
                _SMALL_PIPELINE = _load_pipeline(_SMALL_MODEL_ID)
            _loaded = True
        except Exception as exc:
            _load_error = str(exc)
            print(f"\n[local_inference] ERROR — model failed to load: {exc}\n")
            raise


# ── Minimal OpenAI-compatible response dataclasses ────────────────────────────

@dataclass
class _Message:
    content: str
    role: str = "assistant"


@dataclass
class _Choice:
    message: _Message
    index: int = 0
    finish_reason: str = "stop"


@dataclass
class _Response:
    choices: list[_Choice] = field(default_factory=list)


# ── Client ────────────────────────────────────────────────────────────────────

class LocalChatClient:
    """
    Wraps the globally-loaded pipelines.

    Pass size="large" (default) or size="small" to select which pipeline to use.
    The `model` argument is accepted for backwards compatibility but ignored.
    """

    def __init__(self, model: str = "", size: str = "large", **_ignored):
        self._size = size  # store size; resolve pipeline lazily on first call

    def _pipe(self):
        _ensure_loaded()
        return _SMALL_PIPELINE if self._size == "small" else _LARGE_PIPELINE

    def chat_completion(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.7,
        tools: list | None = None,
        **_ignored,
    ) -> _Response:
        pipe = self._pipe()
        use_sampling = temperature > 0

        # Qwen2.5 eos_token_id is a list [151645, 151643]; pad_token_id must be int
        eos = pipe.tokenizer.eos_token_id
        pad_id = eos[0] if isinstance(eos, (list, tuple)) else eos

        kwargs: dict = dict(
            max_new_tokens=max_tokens,
            do_sample=use_sampling,
            return_full_text=False,
            pad_token_id=pad_id,
            eos_token_id=eos,
        )
        if use_sampling:
            kwargs["temperature"] = temperature

        role_summary = " → ".join(m.get("role", "?") for m in messages)
        print(f"[LLM] calling ({role_summary}) | max_new_tokens={max_tokens} do_sample={use_sampling}")
        output = pipe(messages, **kwargs)
        print(f"[LLM] done")

        generated = output[0]["generated_text"]
        if isinstance(generated, list):
            text = generated[-1].get("content", "") if generated else ""
        else:
            text = str(generated)

        return _Response(choices=[_Choice(message=_Message(content=text))])
