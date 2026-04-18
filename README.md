# mewmo

A local AI memory system with semantic search, content-addressable file storage, and conflict detection. Exposes an MCP server so any MCP-compatible agent (Claude Code, OpenCode, etc.) can store and retrieve knowledge.

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

### Modules

| Module | Role |
|--------|------|
| `config.py` | Runtime config from environment variables |
| `llm.py` | HTTP client for any OpenAI-compatible server |
| `filestore.py` | Content-addressable text storage (SHA256 → `.md`) |
| `sourcestore.py` | Binary source vault (PDF, audio, images) |
| `vectordb.py` | ChromaDB wrapper for semantic search |
| `memory.py` | High-level store/recall/forget API |
| `preprocess.py` | File → text (PDF via pymupdf, audio via faster-whisper, image via vision LLM) |
| `extractor.py` | LLM extracts discrete facts from text |
| `conflicts.py` | JSON store for pending/resolved conflicts |
| `validator.py` | Semantic search + LLM conflict check before storing |
| `ingest.py` | Orchestrates the full pipeline (`Ingester` class) |
| `server.py` | MCP server exposing all operations as tools |

---

## Setup

### 1. Requirements

- [uv](https://docs.astral.sh/uv/)
- A running OpenAI-compatible LLM server — [llama.cpp](https://github.com/ggerganov/llama.cpp), [vLLM](https://github.com/vllm-project/vllm), [LM Studio](https://lmstudio.ai/), etc.

### 2. Install

```bash
git clone <repo>
cd mewmo
uv sync
uv pip install -e .
```

### 3. Configure

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

```env
# URL of your LLM server
MEWMO_LLM_BASE_URL=http://127.0.0.1:8081/v1

# API token — required for vLLM and hosted endpoints.
# For llama.cpp with no auth, use any non-empty string (e.g. "local").
MEWMO_LLM_API_KEY=your-token-here

# Model name exactly as your server reports it
MEWMO_LLM_MODEL=qwen3.5-8b

# Where to persist memory (file vault, vector DB, conflicts log)
MEWMO_DATA_DIR=./data
```

> **vLLM note**: vLLM requires a bearer token set via `--api-key` when starting the server.
> Use the same token as `MEWMO_LLM_API_KEY`.

---

## MCP Server

### Register with Claude Code

Add to `~/.claude.json` (or your project's `.mcp.json`):

```json
{
  "mcpServers": {
    "mewmo": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/mewmo", "run", "mewmo-server"],
      "env": {
        "MEWMO_LLM_BASE_URL": "http://127.0.0.1:8081/v1",
        "MEWMO_LLM_API_KEY":  "your-token-here",
        "MEWMO_LLM_MODEL":    "qwen3.5-8b",
        "MEWMO_DATA_DIR":     "/absolute/path/to/data"
      }
    }
  }
}
```

### Register with OpenCode

Add to your OpenCode config (`~/.config/opencode/config.json`):

```json
{
  "mcp": {
    "mewmo": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/mewmo", "run", "mewmo-server"],
      "env": {
        "MEWMO_LLM_BASE_URL": "http://127.0.0.1:8081/v1",
        "MEWMO_LLM_API_KEY":  "your-token-here",
        "MEWMO_LLM_MODEL":    "qwen3.5-8b",
        "MEWMO_DATA_DIR":     "/absolute/path/to/data"
      }
    }
  }
}
```

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `ingest_file(path)` | Ingest a file (PDF, audio, image, text) |
| `ingest_text(text, source_hint)` | Ingest plain text directly |
| `recall(query, n)` | Semantic search — returns top-n relevant facts |
| `list_conflicts()` | List facts pending user review |
| `resolve_conflict(id, keep)` | Resolve a conflict (`new` / `existing` / `both`) |
| `get_source(source_id)` | Get the path to the original source file (proof) |
| `memory_stats()` | Total facts stored, pending conflicts, data dir |

---

## Python API

```python
from mewmo import Ingester

ing = Ingester("./data")

# Ingest a file
result = ing.ingest_file("health_report.pdf")
# {"facts_extracted": 24, "facts_stored": 22, "conflicts_flagged": 2, ...}

# Ingest plain text
ing.ingest_text("LDL cholesterol should be below 100 mg/dL.", source_hint="manual")

# Semantic search
results = ing.memory.recall("cholesterol levels")
for r in results:
    print(r["content"])
    # r["metadata"] contains source_id, topic, location

# Get original source file as proof
path = ing.get_source_path(results[0]["metadata"]["source_id"])

# Review and resolve conflicts
ing.print_conflicts()
ing.resolve_conflict("<id>", keep="new")       # trust the new fact
ing.resolve_conflict("<id>", keep="existing")  # keep existing
ing.resolve_conflict("<id>", keep="both")      # store both

ing.close()
```

---

## Notes

- **Images**: Requires a vision-capable model (e.g. `qwen2.5-vl`). `qwen3.5-8b` is text-only.
- **Audio**: First run downloads the Whisper `base` model (~150 MB) on demand.
- **Conflicts**: Facts are held back until you resolve them — call `list_conflicts()` after ingestion.
- **Deduplication**: Ingesting the same file twice is a no-op (content-hash check).
- **Backup**: Copy `data/files/`, `data/sources/`, and `data/conflicts.json`. The vector index (`data/chroma/`) is derived and can be rebuilt.
