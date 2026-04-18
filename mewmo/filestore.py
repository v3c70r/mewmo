"""File system layer for storing solid files (content-addressable by SHA256)."""

import hashlib
from pathlib import Path


class FileStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _hash(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _path_for(self, file_id: str) -> Path:
        # 2-level prefix to avoid huge flat directories
        return self.root / file_id[:2] / file_id[2:4] / f"{file_id}.md"

    def write(self, content: str) -> str:
        """Store content, return its hash ID."""
        file_id = self._hash(content)
        path = self._path_for(file_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return file_id

    def read(self, file_id: str) -> str | None:
        """Read content by hash ID. Returns None if not found."""
        path = self._path_for(file_id)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def exists(self, file_id: str) -> bool:
        return self._path_for(file_id).exists()

    def delete(self, file_id: str) -> bool:
        """Delete a file. Returns True if it existed."""
        path = self._path_for(file_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def path_of(self, file_id: str) -> Path:
        """Return the filesystem path for a given ID."""
        return self._path_for(file_id)
