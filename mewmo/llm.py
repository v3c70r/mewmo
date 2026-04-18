"""Thin LLM client for any OpenAI-compatible server (llama.cpp, vLLM, LM Studio, etc.)."""

import re
import json
import httpx

from mewmo.config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, LLM_TIMEOUT


def strip_think(text: str) -> str:
    """Remove Qwen3 <think>...</think> reasoning blocks."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_json(raw: str) -> dict | list:
    """Parse JSON from LLM output, with fallback extraction for wrapped responses."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"(\[.*\]|\{.*\})", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"No valid JSON found in LLM output: {raw[:300]!r}")


def chat(messages: list[dict], temperature: float = 0.0) -> str:
    """Send a chat request, return stripped text content."""
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"}
    with httpx.Client(timeout=LLM_TIMEOUT) as client:
        r = client.post(f"{LLM_BASE_URL}/chat/completions", json=payload, headers=headers)
        r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"]
    return strip_think(raw)
