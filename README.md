# mewmo

A local AI memory system with semantic search, content-addressable file storage, and conflict detection.

## Architecture

```
Any file (PDF / audio / image / text)
    ↓
[Preprocessor]   →  raw text / transcript / description
    ↓
[Extractor LLM]  →  discrete facts  [{fact, topic, location}]
    ↓
[Validator]      →  conflict check against existing memory
    ↓
[Memory]         →  FileStore (solid files) + VectorDB (embeddings)
                     SourceStore (original files kept as proof)
```

### Components

| Module | Role |
|--------|------|
| `filestore.py` | Content-addressable text storage (SHA256 → `.md`) |
| `sourcestore.py` | Binary source vault (PDF, audio, images) |
| `vectordb.py` | ChromaDB wrapper for semantic search |
| `memory.py` | High-level store/recall/forget API |
| `preprocess.py` | File → text (PDF via pymupdf, audio via faster-whisper, image via vision LLM) |
| `extractor.py` | LLM extracts discrete facts from text |
| `conflicts.py` | JSON store for pending/resolved conflicts |
| `validator.py` | Semantic search + LLM conflict check before storing |
| `ingest.py` | Orchestrates the full pipeline (`Ingester` class) |
| `llm.py` | HTTP client for local llama.cpp server (OpenAI-compatible) |

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a local [llama.cpp](https://github.com/ggerganov/llama.cpp) server.

```bash
uv sync
uv pip install -e .
```

Configure `mewmo/llm.py` for your local LLM server:
```python
BASE_URL = "http://127.0.0.1:8081/v1"
MODEL = "qwen3.5-8b"
```

## Usage

```python
from mewmo import Ingester

ing = Ingester("./data")

# Ingest a file
result = ing.ingest_file("health_report.pdf")
print(result)
# {"facts_extracted": 24, "facts_stored": 22, "conflicts_flagged": 2, ...}

# Ingest plain text
ing.ingest_text("LDL cholesterol should be below 100 mg/dL.", source_hint="manual")

# Semantic search
results = ing.memory.recall("cholesterol levels")
for r in results:
    print(r["content"])
    print(r["metadata"])  # includes source_id, topic, location

# Get original source file as proof
path = ing.get_source_path(results[0]["metadata"]["source_id"])

# Review and resolve conflicts
ing.print_conflicts()
ing.resolve_conflict("<conflict_id>", keep="new")       # trust new fact
ing.resolve_conflict("<conflict_id>", keep="existing")  # keep existing
ing.resolve_conflict("<conflict_id>", keep="both")      # store both
```

## Notes

- **Images**: Requires a vision-capable model (e.g. `qwen2.5-vl`). `qwen3.5-8b` is text-only.
- **Audio**: First run downloads the Whisper `base` model (~150 MB).
- **Conflicts**: Stored in `data/conflicts.json`. Facts are not stored until conflicts are resolved.
- **Deduplication**: Same file ingested twice is a no-op (content-hash check).
