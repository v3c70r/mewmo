"""Ingestion pipeline: file → preprocess → extract facts → validate → store."""

import hashlib
import logging
from pathlib import Path

from mewmo.memory import Memory
from mewmo.sourcestore import SourceStore
from mewmo.preprocess import preprocess
from mewmo.extractor import extract_facts
from mewmo.conflicts import ConflictStore
from mewmo.validator import Validator

log = logging.getLogger(__name__)


class Ingester:
    def __init__(
        self,
        data_dir: str | Path = "./data",
        conflict_threshold: float = 0.25,
    ):
        data_dir = Path(data_dir)
        self.memory = Memory(data_dir)
        self.sources = SourceStore(data_dir / "sources")
        self.conflicts = ConflictStore(data_dir / "conflicts.json")
        self.validator = Validator(self.memory, self.conflicts, conflict_threshold)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_file(self, path: str | Path) -> dict:
        """
        Full pipeline for a file on disk.
        Returns summary dict with source_id, counts, and flagged conflicts.
        """
        path = Path(path)
        data = path.read_bytes()
        extension = path.suffix

        # Skip if already ingested (deduplication gate)
        source_id = hashlib.sha256(data).hexdigest()
        if self.sources.exists(source_id):
            log.info("Source already ingested: %s", source_id[:12])
            return {"source_id": source_id, "skipped": True,
                    "facts_stored": 0, "conflicts_flagged": 0, "conflicts": []}

        # 1. Store raw source file
        self.sources.write(data, extension)

        # 2. Convert to text
        text = preprocess(data, extension)

        return self._process_text(text, source_id, source_hint=path.name)

    def ingest_text(self, text: str, source_hint: str = "direct input") -> dict:
        """
        Ingest plain text directly (no file needed).
        source_hint is a label used for provenance in metadata.
        """
        source_id = hashlib.sha256(text.encode()).hexdigest()
        return self._process_text(text, source_id, source_hint=source_hint)

    def _process_text(self, text: str, source_id: str, source_hint: str) -> dict:
        """Extract facts, validate against memory, and store. Returns summary."""
        facts = extract_facts(text, source_hint=source_hint)
        log.info("Extracted %d facts from %r", len(facts), source_hint)

        stored, flagged_list = 0, []
        for f in facts:
            result = self.validator.validate_and_store(
                fact=f["fact"],
                topic=f["topic"],
                location=f["location"],
                source_id=source_id,
            )
            if result is not None:
                stored += 1
            else:
                flagged_list.append(f["fact"])

        return {
            "source_id": source_id,
            "skipped": False,
            "facts_extracted": len(facts),
            "facts_stored": stored,
            "conflicts_flagged": len(flagged_list),
            "conflicts": flagged_list,
        }

    # ------------------------------------------------------------------
    # Conflict review
    # ------------------------------------------------------------------

    def review_conflicts(self) -> list[dict]:
        """Return all pending conflicts."""
        return self.conflicts.pending()

    def print_conflicts(self):
        """Print pending conflicts in a readable format."""
        pending = self.conflicts.pending()
        if not pending:
            print("No pending conflicts.")
            return
        print(f"\n{len(pending)} conflict(s) pending review:\n")
        for c in pending:
            print(f"  ID       : {c['id']}")
            print(f"  Topic    : {c['topic']}")
            print(f"  Existing : {c['existing_fact']}")
            print(f"  New      : {c['new_fact']}")
            print(f"  Likely   : {c['likely_true']}")
            print(f"  Reason   : {c['reasoning']}")
            print()

    def resolve_conflict(self, conflict_id: str, keep: str) -> bool:
        """
        Resolve a pending conflict.

        keep:
          'new'      — discard existing, store the new fact
          'existing' — keep existing as-is, discard new fact
          'both'     — store new fact alongside existing (no winner)

        Returns True if the conflict was found and resolved.
        """
        pending = {c["id"]: c for c in self.conflicts.pending()}
        if conflict_id not in pending:
            return False

        entry = pending[conflict_id]

        if keep in ("new", "both"):
            self.memory.store(entry["new_fact"], {
                "topic": entry["topic"],
                "location": entry["location"],
                "source_id": entry["source_id"],
            })
        if keep == "new":
            # Remove the superseded existing fact
            self.memory.forget(entry["existing_fact_id"])

        resolution_label = {"new": "kept_new", "existing": "kept_existing", "both": "kept_both"}
        self.conflicts.resolve(conflict_id, resolution_label.get(keep, keep))
        return True

    # ------------------------------------------------------------------
    # Source lookup (proof)
    # ------------------------------------------------------------------

    def get_source_path(self, source_id: str) -> Path | None:
        """Return filesystem path of the original source file."""
        return self.sources.find_path(source_id)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        self.memory.close()
