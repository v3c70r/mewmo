"""LLM-based fact extraction: text → list of discrete facts."""

import logging
from mewmo.llm import chat, parse_json

log = logging.getLogger(__name__)

_SYSTEM = """\
You are a fact extraction engine.
Extract every discrete, self-contained factual claim from the provided text.

Return ONLY a JSON array. Each element must have exactly these keys:
- "fact": the factual statement as a single clear sentence
- "topic": a short topic label (e.g. "cholesterol", "blood pressure", "medication")
- "location": source location (e.g. "Page 3", "00:42", "paragraph 2"), or null

Rules:
- Each fact must be understandable without surrounding context
- Include relevant numbers, units, thresholds, and names
- Do not include opinions, recommendations, or uncertain claims as facts
- Do not add any explanation or markdown — return only the JSON array"""


def extract_facts(text: str, source_hint: str = "") -> list[dict]:
    """
    Run LLM extraction on preprocessed text.
    Returns list of {"fact": str, "topic": str, "location": str | None}.
    Logs a warning and returns [] if the LLM returns nothing usable.
    """
    user_content = f"Source: {source_hint}\n\n{text}" if source_hint else text
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user_content},
    ]
    raw = chat(messages, temperature=0.0)

    try:
        facts = parse_json(raw)
    except (ValueError, Exception) as e:
        log.warning("Extractor failed to parse JSON: %s | raw: %.200s", e, raw)
        return []

    if not isinstance(facts, list):
        log.warning("Extractor returned non-list: %r", facts)
        return []

    required = {"fact", "topic"}
    valid = []
    for f in facts:
        if not isinstance(f, dict) or not required.issubset(f.keys()):
            continue
        valid.append({
            "fact": str(f["fact"]),
            "topic": str(f["topic"]),
            "location": str(f["location"]) if f.get("location") else "",
        })

    if not valid:
        log.warning("Extractor returned 0 valid facts from source: %r", source_hint)
    return valid
