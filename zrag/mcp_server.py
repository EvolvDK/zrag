"""
MCP (Model Context Protocol) Server for zrag.

Exposes the local zrag daemon capabilities to AI assistants (Claude Desktop, Cursor, etc.).

Supports two transports:
  - stdio: for Claude Desktop, Cursor, and other desktop AI tools
  - streamable-http: for browser-based UIs (llama.cpp, etc.) and HTTP clients
"""

from typing import List, Dict, Any

import anyio
import uvicorn
from starlette.middleware.cors import CORSMiddleware

from mcp.server.fastmcp import FastMCP
from zrag.cli import ensure_daemon

MAX_PREVIEW_CHARS = 300

def _format_results(results: List[Dict[str, Any]], *, full_content: bool = False) -> str:
    if not results:
        return "No results found."
    lines = []
    for i, r in enumerate(results, 1):
        score = r.get("score")
        score_str = f", Score: {score:.4f}" if score is not None else ""
        fields = r.get("fields", {})
        source = fields.get("source", "unknown")
        context = fields.get("context", "")
        content = fields.get("content", "")

        lines.append(f"--- Result {i} (Source: {source}{score_str}) ---")
        if context:
            lines.append(f"Context: {context}")

        if full_content:
            lines.append(f"Content:\n{content}\n")
        else:
            preview = content[:MAX_PREVIEW_CHARS].replace('\n', ' ')
            if len(content) > MAX_PREVIEW_CHARS:
                preview += "..."
            lines.append(f"Preview: {preview}\n")

    return "\n".join(lines)

def create_mcp_server(host: str = "127.0.0.1", port: int = 8000) -> FastMCP:
    mcp = FastMCP("zrag", host=host, port=port)

    def _client():
        """Shortcut for MCP tools: fail fast (10s) if daemon isn't ready."""
        return ensure_daemon(quiet=True, raise_errors=True, max_wait_seconds=120.0)

    @mcp.tool()
    def search(query: str, collection: str = "default", top_k: int = 5, full_content: bool = False) -> str:
        """Perform a BM25 keyword search. Returns content previews by default; set full_content=True for complete content."""
        with _client() as client:
            res = client.post("/search/bm25", json={"collection_name": collection, "query": query, "top_k": top_k})
            return _format_results(res.get("results",[]), full_content=full_content)

    @mcp.tool()
    def vsearch(query: str, collection: str = "default", top_k: int = 5, full_content: bool = False) -> str:
        """Perform a dense vector semantic search. Returns content previews by default; set full_content=True for complete content."""
        with _client() as client:
            res = client.post("/search/vector", json={"collection_name": collection, "query": query, "top_k": top_k})
            return _format_results(res.get("results",[]), full_content=full_content)

    @mcp.tool()
    def hybrid_query(query: str, collection: str = "default", top_k: int = 5, full_content: bool = False) -> str:
        """Perform a hybrid (BM25 + Vector + RRF) semantic search. Returns content previews by default; set full_content=True for complete content."""
        with _client() as client:
            res = client.post("/search/hybrid", json={"collection_name": collection, "query": query, "top_k": top_k})
            return _format_results(res.get("results",[]), full_content=full_content)

    @mcp.tool()
    def retrieve_lines(identifier: str, collection: str = "default", limit: int = 100) -> str:
        """Get full document or specific lines with complete content.
        Format: 'doc_id', 'glob:pattern', or 'path:start_line-end_line'.
        """
        with _client() as client:
            res = client.post("/get", json={"collection_name": collection, "identifier": identifier, "format": "json", "limit": limit})
            if isinstance(res, list):
                return _format_results(res, full_content=True)
            return _format_results([res], full_content=True)

    @mcp.tool()
    def list_collections() -> str:
        """List all available document collections."""
        with _client() as client:
            res = client.get("/collections")
            if not res:
                return "No collections found."
            lines = ["Collections:"]
            for c in res:
                lines.append(f"- {c['name']} ({c['document_count']} docs, {c['size_bytes']} bytes)")
            return "\n".join(lines)

    @mcp.tool()
    def get_status() -> str:
        """Get daemon status including collections loaded and model state."""
        with _client() as client:
            res = client.get("/status")
            models = "Yes" if res.get("models_loaded") else "No"
            return f"Daemon: running\nCollections loaded: {res.get('collections_loaded', 0)}\nModels loaded: {models}"

    @mcp.tool()
    def collection_inspect(name: str) -> str:
        """Inspect a collection's schema: fields, vector dimensions, and index types."""
        with _client() as client:
            schema = client.get(f"/collections/{name}/inspect")
            lines = [f"Collection: {schema['name']}", "", "Fields:"]
            for f in schema.get("fields", []):
                lines.append(f"  - {f['name']}: {f['data_type']}")
            lines.append("")
            lines.append("Vectors:")
            for v in schema.get("vectors", []):
                dim = f", dimension={v['dimension']}" if v.get("dimension") else ""
                idx = f", index={v['index_type']}" if v.get("index_type") else ""
                lines.append(f"  - {v['name']}: {v['data_type']}{dim}{idx}")
            return "\n".join(lines)

    @mcp.tool()
    def collection_list_files(name: str, filter: str = "") -> str:
        """List all files in a collection, optionally filtered by a glob pattern."""
        with _client() as client:
            params = {}
            if filter:
                params["filter"] = filter
            files = client.get(f"/collections/{name}/files", params=params)
            if not files:
                return "No files found."
            lines = [f"Files in '{name}':"]
            for f in files:
                lines.append(f"  {f['source']} ({f['file_type']}, {f['chunk_count']} chunks)")
            return "\n".join(lines)

    @mcp.tool()
    def list_contexts(collection_name: str = "") -> str:
        """List all context descriptions, optionally filtered by collection."""
        with _client() as client:
            params = {}
            if collection_name:
                params["collection_name"] = collection_name
            res = client.get("/context/list", params=params)
            if not res:
                return "No contexts found."
            lines = ["Contexts:"]
            for ctx in res:
                lines.append(f"  {ctx['path']}: {ctx['description']}")
            return "\n".join(lines)

    @mcp.tool()
    def context_add(path: str, description: str) -> str:
        """Add or update a context description for a given path."""
        with _client() as client:
            res = client.post("/context/add", json={"path": path, "description": description})
            return f"Context added/updated for {res['path']}: {res['description']}"

    @mcp.tool()
    def context_remove(path: str) -> str:
        """Remove context(s) matching a path or glob pattern."""
        with _client() as client:
            res = client.delete("/context/remove", params={"path": path})
            return f"Removed {res.get('count', 'matching context(s)')}"

    @mcp.tool()
    def ingest_file(collection_name: str, filepath: str, source: str = "") -> str:
        """Ingest a local file into a collection."""
        with _client() as client:
            body = {"collection_name": collection_name, "filepath": filepath}
            if source:
                body["source"] = source
            res = client.post("/ingest/file", json=body)
            errors = res.get("errors", [])
            parts = [f"added={res.get('added', 0)}, updated={res.get('updated', 0)}"]
            if errors:
                parts.append(f"errors={errors}")
            return f"Ingested '{filepath}': " + ", ".join(parts)

    @mcp.tool()
    def update_collection(name: str, pull: bool = False) -> str:
        """Update a collection by scanning for filesystem changes. Optionally run git pull first."""
        with _client() as client:
            res = client.post(f"/collections/{name}/update", params={"pull": pull})
            t = res.get("elapsed", 0)
            errors = res.get("errors", [])
            parts = [f"added={res.get('added', 0)}", f"updated={res.get('updated', 0)}", f"removed={res.get('removed', 0)}", f"took {t:.1f}s"]
            if errors:
                parts.append(f"errors={errors}")
            return f"Updated '{name}': " + ", ".join(parts)

    return mcp

def run_stdio():
    """Run MCP server over stdio (for Claude Desktop, Cursor, etc.)."""
    mcp = create_mcp_server()
    mcp.run(transport="stdio")

def run_http(port: int = 8002):
    """Run MCP server over Streamable HTTP (for browser UIs, llama.cpp, etc.)."""
    mcp = create_mcp_server(host="0.0.0.0", port=port)
    app = mcp.streamable_http_app()
    app = CORSMiddleware(
        app,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["Mcp-Session-Id"],
    )
    config = uvicorn.Config(app, host="0.0.0.0", port=port)
    server = uvicorn.Server(config)
    anyio.run(server.serve)