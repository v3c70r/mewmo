"""Memory system tying FileStore and VectorDB together."""

from pathlib import Path
from mewmo.filestore import FileStore
from mewmo.vectordb import VectorDB


class Memory:
    def __init__(self, data_dir: str | Path = "./data"):
        data_dir = Path(data_dir)
        self.files = FileStore(data_dir / "files")
        self.vectors = VectorDB(data_dir / "chroma")

    def store(self, content: str, metadata: dict | None = None) -> str:
        """Store a memory: writes file + indexes embedding. Returns file ID."""
        file_id = self.files.write(content)
        self.vectors.add(file_id, content, metadata)
        return file_id

    def recall(self, query: str, n: int = 5) -> list[dict]:
        """Semantic search across memories. Returns results with full content."""
        results = self.vectors.search(query, n_results=n)
        for r in results:
            r["content"] = self.files.read(r["id"]) or r["content"]
        return results

    def forget(self, file_id: str) -> bool:
        """Remove a memory from both stores."""
        self.vectors.delete(file_id)
        return self.files.delete(file_id)

    def count(self) -> int:
        return self.vectors.count()

    def close(self):
        """Release resources (important on Windows)."""
        self.vectors.close()
