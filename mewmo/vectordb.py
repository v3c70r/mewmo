"""Vector DB layer wrapping ChromaDB with local sentence-transformer embeddings."""

import chromadb
from pathlib import Path


class VectorDB:
    def __init__(self, persist_dir: Path, collection_name: str = "memories"):
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, file_id: str, content: str, metadata: dict | None = None):
        """Index a document. file_id is the content-hash from FileStore."""
        kwargs: dict = {"ids": [file_id], "documents": [content]}
        if metadata:
            kwargs["metadatas"] = [metadata]
        self.collection.upsert(**kwargs)

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        """Semantic search. Returns list of {id, content, metadata, distance}."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
        )
        out = []
        for i in range(len(results["ids"][0])):
            out.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return out

    def delete(self, file_id: str):
        """Remove a document from the index."""
        self.collection.delete(ids=[file_id])

    def count(self) -> int:
        return self.collection.count()

    def close(self):
        """Release file handles (important on Windows)."""
        self.client._system.stop()
