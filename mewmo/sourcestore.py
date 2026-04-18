"""Vault for original source files (PDFs, audio, images, etc.).

Files are stored content-addressably by SHA256 hash, preserving their extension.
This is separate from FileStore which stores extracted text (solid files).
"""

import hashlib
from pathlib import Path


class SourceStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _hash(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _prefix_dir(self, source_id: str) -> Path:
        return self.root / source_id[:2] / source_id[2:4]

    def write(self, data: bytes, extension: str) -> str:
        """Store raw bytes. Returns source ID (SHA256 hash). Idempotent."""
        source_id = self._hash(data)
        ext = extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        path = self._prefix_dir(source_id) / f"{source_id}{ext}"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        return source_id

    def find_path(self, source_id: str) -> Path | None:
        """Find the stored file path (any extension). Returns None if not found."""
        prefix_dir = self._prefix_dir(source_id)
        if not prefix_dir.exists():
            return None
        matches = list(prefix_dir.glob(f"{source_id}.*"))
        return matches[0] if matches else None

    def read(self, source_id: str) -> tuple[bytes, str] | None:
        """Returns (bytes, extension) or None."""
        path = self.find_path(source_id)
        if path is None:
            return None
        return path.read_bytes(), path.suffix

    def exists(self, source_id: str) -> bool:
        return self.find_path(source_id) is not None

    def delete(self, source_id: str) -> bool:
        path = self.find_path(source_id)
        if path:
            path.unlink()
            return True
        return False
