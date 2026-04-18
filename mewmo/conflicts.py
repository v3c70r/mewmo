"""Conflict store: tracks contradictions between new and existing facts.

Stored as a JSON file (pre-relational DB). Atomic writes via temp+rename.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


class ConflictStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        if self.path.exists():
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self._data = {"pending": [], "resolved": []}
            self._save()

    def _save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def add(
        self,
        new_fact: str,
        existing_fact: str,
        existing_fact_id: str,
        source_id: str,
        topic: str,
        location: str,
        likely_true: str,   # "new" | "existing" | "ambiguous"
        reasoning: str,
        new_fact_id: str | None = None,
    ) -> str:
        """Record a detected conflict. Returns conflict ID."""
        entry = {
            "id": str(uuid.uuid4())[:8],
            "detected_at": datetime.now(timezone.utc).isoformat(),
            "new_fact": new_fact,
            "new_fact_id": new_fact_id,
            "existing_fact": existing_fact,
            "existing_fact_id": existing_fact_id,
            "source_id": source_id,
            "topic": topic,
            "location": location,
            "likely_true": likely_true,
            "reasoning": reasoning,
        }
        self._data["pending"].append(entry)
        self._save()
        return entry["id"]

    def resolve(self, conflict_id: str, resolution: str) -> bool:
        """
        Mark a conflict resolved.
        resolution: 'kept_new' | 'kept_existing' | 'kept_both'
        Returns True if found, False if not found.
        """
        for i, entry in enumerate(self._data["pending"]):
            if entry["id"] == conflict_id:
                entry["resolution"] = resolution
                entry["resolved_at"] = datetime.now(timezone.utc).isoformat()
                self._data["resolved"].append(entry)
                self._data["pending"].pop(i)
                self._save()
                return True
        return False

    def pending(self) -> list[dict]:
        return list(self._data["pending"])

    def resolved(self) -> list[dict]:
        return list(self._data["resolved"])

    def pending_count(self) -> int:
        return len(self._data["pending"])
