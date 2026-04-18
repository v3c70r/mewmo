"""MCP server exposing mewmo memory operations as tools.

Run with:
    uv run mewmo-server

Or register in your MCP client config:
    {
      "mcpServers": {
        "mewmo": {
          "command": "uv",
          "args": ["--directory", "/path/to/mewmo", "run", "mewmo-server"],
          "env": {
            "MEWMO_LLM_BASE_URL": "http://127.0.0.1:8081/v1",
            "MEWMO_LLM_API_KEY":  "your-token-here",
            "MEWMO_LLM_MODEL":    "qwen3.5-8b",
            "MEWMO_DATA_DIR":     "/path/to/your/data"
          }
        }
      }
    }
"""

import json
from mcp.server.fastmcp import FastMCP
from mewmo.config import DATA_DIR, CONFLICT_THRESHOLD
from mewmo.ingest import Ingester

mcp = FastMCP(
    "mewmo",
    instructions=(
        "mewmo is a persistent AI memory system. "
        "Use it to store knowledge from files and text, search memories semantically, "
        "and review conflicting facts before storing them."
    ),
)

# Single shared Ingester instance for the lifetime of the server
_ingester: Ingester | None = None


def get_ingester() -> Ingester:
    global _ingester
    if _ingester is None:
        _ingester = Ingester(DATA_DIR, conflict_threshold=CONFLICT_THRESHOLD)
    return _ingester


# ------------------------------------------------------------------
# Tools
# ------------------------------------------------------------------

@mcp.tool()
def ingest_file(path: str) -> str:
    """
    Ingest a file into memory (PDF, audio, image, or text).

    Preprocesses the file, extracts discrete facts via LLM, checks each fact
    against existing memory for conflicts, and stores non-conflicting facts.

    Args:
        path: Absolute or relative path to the file on disk.

    Returns:
        JSON summary: facts_extracted, facts_stored, conflicts_flagged, source_id.
    """
    result = get_ingester().ingest_file(path)
    return json.dumps(result, indent=2)


@mcp.tool()
def ingest_text(text: str, source_hint: str = "direct input") -> str:
    """
    Ingest plain text directly into memory.

    Args:
        text: The text content to extract facts from.
        source_hint: A label for provenance (e.g. "user note", "web page").

    Returns:
        JSON summary: facts_extracted, facts_stored, conflicts_flagged, source_id.
    """
    result = get_ingester().ingest_text(text, source_hint=source_hint)
    return json.dumps(result, indent=2)


@mcp.tool()
def recall(query: str, n: int = 5) -> str:
    """
    Search memory semantically and return the most relevant facts.

    Args:
        query: Natural language search query.
        n:     Number of results to return (default 5).

    Returns:
        JSON array of matches, each with: content, metadata (topic, source_id,
        location), and distance score (lower = more similar).
    """
    results = get_ingester().memory.recall(query, n=n)
    # Strip heavy fields for readability; keep content + key metadata
    slim = [
        {
            "content":   r["content"],
            "topic":     r["metadata"].get("topic", ""),
            "location":  r["metadata"].get("location", ""),
            "source_id": r["metadata"].get("source_id", ""),
            "distance":  round(r["distance"], 4),
        }
        for r in results
    ]
    return json.dumps(slim, indent=2)


@mcp.tool()
def list_conflicts() -> str:
    """
    List all pending conflicts that need user review.

    A conflict occurs when a newly ingested fact contradicts an existing memory.
    Both facts are shown along with the LLM's reasoning and its recommendation
    for which is more likely to be true.

    Returns:
        JSON array of pending conflicts, each with: id, existing_fact, new_fact,
        likely_true ('new'|'existing'|'ambiguous'), reasoning, topic, detected_at.
    """
    pending = get_ingester().review_conflicts()
    if not pending:
        return "[]"
    return json.dumps(pending, indent=2)


@mcp.tool()
def resolve_conflict(conflict_id: str, keep: str) -> str:
    """
    Resolve a pending conflict by deciding which fact to keep.

    Args:
        conflict_id: The short ID from list_conflicts (e.g. 'a3f1bc2e').
        keep:        One of:
                       'new'      — store the new fact, remove the existing one
                       'existing' — keep existing fact as-is, discard new
                       'both'     — store both facts (no contradiction assumed)

    Returns:
        Confirmation message or error if conflict_id not found.
    """
    ok = get_ingester().resolve_conflict(conflict_id, keep)
    if ok:
        return f"Conflict {conflict_id!r} resolved: keep={keep!r}."
    return f"Conflict {conflict_id!r} not found in pending list."


@mcp.tool()
def get_source(source_id: str) -> str:
    """
    Return the filesystem path of the original source file for a given memory.

    Useful for retrieving proof — e.g. the original PDF a fact was extracted from.

    Args:
        source_id: The source_id from a recall() result's metadata.

    Returns:
        Absolute path to the source file, or a not-found message.
    """
    path = get_ingester().get_source_path(source_id)
    if path:
        return str(path.resolve())
    return f"Source {source_id!r} not found in vault."


@mcp.tool()
def memory_stats() -> str:
    """
    Return a summary of the current memory state.

    Returns:
        JSON with total facts stored, pending conflicts count.
    """
    ing = get_ingester()
    return json.dumps({
        "facts_stored":       ing.memory.count(),
        "conflicts_pending":  ing.conflicts.pending_count(),
        "data_dir":           str(DATA_DIR.resolve()),
    }, indent=2)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
