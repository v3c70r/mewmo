"""Thin LLM client for the local llama.cpp server (OpenAI-compatible)."""

import re
import json
import httpx

BASE_URL = "http://127.0.0.1:8081/v1"
MODEL = "qwen3.5-8b"
TIMEOUT = 120.0  # seconds; long enough for large documents


def strip_think(text: str) -> str:
    """Remove Qwen3 <think>...</think> reasoning blocks."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_json(raw: str) -> dict | list:
    """Parse JSON from LLM output, with fallback extraction for wrapped responses."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to find a JSON object or array embedded in surrounding text
        match = re.search(r"(\[.*\]|\{.*\})", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"No valid JSON found in LLM output: {raw[:300]!r}")


def chat(messages: list[dict], temperature: float = 0.0) -> str:
    """Send a chat request, return stripped text content."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    with httpx.Client(timeout=TIMEOUT) as client:
        r = client.post(f"{BASE_URL}/chat/completions", json=payload)
        r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"]
    return strip_think(raw)
