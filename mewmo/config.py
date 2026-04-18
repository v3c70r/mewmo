"""Runtime configuration loaded from environment variables.

Copy .env.example to .env and fill in your values, or set env vars directly.
"""

import os
from pathlib import Path


def _require(key: str) -> str:
    val = os.environ.get(key, "").strip()
    if not val:
        raise RuntimeError(
            f"Required environment variable {key!r} is not set. "
            "See .env.example for setup instructions."
        )
    return val


def _optional(key: str, default: str) -> str:
    return os.environ.get(key, "").strip() or default


# LLM endpoint (OpenAI-compatible: llama.cpp, vLLM, LM Studio, etc.)
LLM_BASE_URL: str = _optional("MEWMO_LLM_BASE_URL", "http://127.0.0.1:8081/v1")
LLM_API_KEY: str  = _optional("MEWMO_LLM_API_KEY",  "local")
LLM_MODEL: str    = _optional("MEWMO_LLM_MODEL",    "qwen3.5-8b")
LLM_TIMEOUT: float = float(_optional("MEWMO_LLM_TIMEOUT", "120"))

# Data directory for file store, vector DB, source vault, conflicts
DATA_DIR: Path = Path(_optional("MEWMO_DATA_DIR", "./data"))

# Conflict detection sensitivity (cosine distance 0–1; lower = stricter)
CONFLICT_THRESHOLD: float = float(_optional("MEWMO_CONFLICT_THRESHOLD", "0.25"))
