"""Conflict detection: checks new facts against existing memory before storing."""

import logging
from mewmo.llm import chat, parse_json
from mewmo.conflicts import ConflictStore

log = logging.getLogger(__name__)

_SYSTEM = """\
You are a fact conflict detector.
Given two factual claims, determine if they contradict each other.

Return ONLY a JSON object with exactly these keys:
- "contradicts": true or false
- "likely_true": one of "new", "existing", or "ambiguous"
- "reasoning": one sentence explaining your decision

Consider: if both facts could be simultaneously true, they do not contradict."""


class Validator:
    def __init__(
        self,
        memory,           # Memory instance
        conflicts: ConflictStore,
        distance_threshold: float = 0.25,
        top_k: int = 5,
    ):
        self.memory = memory
        self.conflicts = conflicts
        self.threshold = distance_threshold
        self.top_k = top_k

    def validate_and_store(
        self,
        fact: str,
        topic: str,
        location: str,
        source_id: str,
        metadata: dict | None = None,
    ) -> str | None:
        """
        Search for semantically close existing memories and run conflict check.

        - If no conflict found: stores fact and returns its file_id.
        - If conflict found: records it in ConflictStore, does NOT store, returns None.
        """
        n = min(self.top_k, self.memory.count())
        if n == 0:
            # Nothing in memory yet — store directly
            return self._store(fact, topic, location, source_id, metadata)

        nearby = self.memory.recall(fact, n=n)
        for candidate in nearby:
            if candidate["distance"] > self.threshold:
                continue  # semantically too far apart

            result = self._check_conflict(fact, candidate["content"])
            if result.get("contradicts"):
                conflict_id = self.conflicts.add(
                    new_fact=fact,
                    existing_fact=candidate["content"],
                    existing_fact_id=candidate["id"],
                    source_id=source_id,
                    topic=topic,
                    location=location,
                    likely_true=result.get("likely_true", "ambiguous"),
                    reasoning=result.get("reasoning", ""),
                )
                log.info("Conflict flagged [%s]: %s", conflict_id, fact[:80])
                return None  # flagged, not stored

        return self._store(fact, topic, location, source_id, metadata)

    def _store(self, fact, topic, location, source_id, extra_meta):
        meta = {"topic": topic, "location": location, "source_id": source_id}
        if extra_meta:
            meta.update(extra_meta)
        return self.memory.store(fact, meta)

    def _check_conflict(self, new_fact: str, existing_fact: str) -> dict:
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": (
                f"New fact: {new_fact}\n"
                f"Existing fact: {existing_fact}"
            )},
        ]
        raw = chat(messages, temperature=0.0)
        try:
            return parse_json(raw)
        except Exception as e:
            log.warning("Conflict check parse error: %s | raw: %.200s", e, raw)
            return {"contradicts": False, "likely_true": "ambiguous", "reasoning": str(e)}
