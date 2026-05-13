"""
CLI client for zrag.

Provides command-line interface that communicates with the local daemon.
"""

import sys
import time
import httpx
from pathlib import Path
from typing import Optional, List, Dict, Any
import click

from rich.console import Console
from rich.table import Table
from rich import print as rprint

from zrag.formatters import OutputFormat, OutputFormatter


DAEMON_HOST = "127.0.0.1"
DAEMON_PORT = 8765
DAEMON_URL = f"http://{DAEMON_HOST}:{DAEMON_PORT}"

console = Console()


# =========================================================================
# Client & Connection Utilities
# =========================================================================

class DaemonClient:
    """HTTP client context manager for cleanly communicating with the daemon."""
    
    def __init__(self, base_url: str = DAEMON_URL, raise_errors: bool = False, quiet: bool = False):
        self.base_url = base_url
        self.client = httpx.Client(timeout=120.0)
        self.raise_errors = raise_errors
        self.quiet = quiet
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()

    def _handle_response(self, response: httpx.Response) -> dict:
        try:
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            detail = e.response.json().get("detail", str(e))
            if self.raise_errors:
                raise RuntimeError(detail)
            if not self.quiet:
                console.print(f"[red]Error:[/red] {detail}")
            sys.exit(1)
        except Exception as e:
            if self.raise_errors:
                raise RuntimeError(str(e))
            if not self.quiet:
                console.print(f"[red]Unexpected Connection Error:[/red] {e}")
            sys.exit(1)

    def get(self, endpoint: str, **kwargs) -> dict:
        return self._handle_response(self.client.get(f"{self.base_url}{endpoint}", **kwargs))
    
    def post(self, endpoint: str, **kwargs) -> dict:
        return self._handle_response(self.client.post(f"{self.base_url}{endpoint}", **kwargs))
    
    def delete(self, endpoint: str, **kwargs) -> dict:
        return self._handle_response(self.client.delete(f"{self.base_url}{endpoint}", **kwargs))


def check_daemon() -> bool:
    try:
        import socket
        with socket.create_connection((DAEMON_HOST, DAEMON_PORT), timeout=1.0):
            return True
    except OSError:
        return False


def ensure_daemon(quiet: bool = False, raise_errors: bool = False, max_wait_seconds: float = 120.0) -> DaemonClient:
    if not check_daemon():
        if not quiet:
            console.print("[yellow]Daemon is not running. Auto-starting...[/yellow]")
        import subprocess

        cmd =[
            sys.executable, "-c",
            f"from zrag.daemon import run_daemon; run_daemon(host='{DAEMON_HOST}', port={DAEMON_PORT})"
        ]
        subprocess.Popen(
            cmd,
            stdout=open('/tmp/zrag_daemon.log', 'w'),
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True
        )
        
        deadline = time.time() + max_wait_seconds
        while time.time() < deadline:
            if check_daemon():
                time.sleep(0.5)
                return DaemonClient(raise_errors=raise_errors, quiet=quiet)
            time.sleep(0.1)

        if not quiet:
            console.print(f"[red]Error: Timeout after {max_wait_seconds:.0f}s waiting for daemon to start.[/red]")
        if raise_errors:
            raise RuntimeError(f"Timeout after {max_wait_seconds:.0f}s waiting for daemon to start.")
        sys.exit(1)

    return DaemonClient(raise_errors=raise_errors, quiet=quiet)

# =========================================================================
# Shared CLI Helpers
# =========================================================================

_search_options =[
    click.option("-c", "--collection", help="Collection name"),
    click.option("-n", "--top-k", type=int, help="Number of results"),
    click.option("--filter", "filter_expr", help="Filter expression"),
    click.option("--threshold", type=float, help="Minimum score threshold"),
    click.option("--all", "return_all", is_flag=True, help="Return all results passing threshold"),
    click.option("--full", is_flag=True, help="Show full content"),
    click.option("--line-numbers", is_flag=True, help="Show line numbers"),
    click.option("--files", "files_only", is_flag=True, help="Show only file information"),
    click.option("--json", "fmt_json", is_flag=True, help="Output as JSON"),
    click.option("--csv", "fmt_csv", is_flag=True, help="Output as CSV"),
    click.option("--md", "fmt_md", is_flag=True, help="Output as Markdown"),
    click.option("--xml", "fmt_xml", is_flag=True, help="Output as XML"),
    click.option("--explain", is_flag=True, help="Explain query expansions, RRF, and reranking"),
]

def search_options(func):
    """Decorator to inject standard search formatting flags."""
    for option in reversed(_search_options):
        func = option(func)
    return func


def print_search_results(
    results: List[Dict[str, Any]], elapsed: float, title: str,
    explain_data: Optional[Dict[str, Any]], threshold: Optional[float], full: bool, line_numbers: bool,
    files_only: bool, fmt_json: bool, fmt_csv: bool, fmt_md: bool, fmt_xml: bool
):
    """Unified handler for filtering and printing output formats."""
    
    # 1. Apply Threshold Filtering
    if threshold is not None:
        results =[r for r in results if r.get("score", 0.0) >= threshold]

    # 2. Handle standard structured formats using the OutputFormatter
    if fmt_json or fmt_csv or fmt_md or fmt_xml:
        fmt = OutputFormat.JSON if fmt_json else OutputFormat.CSV if fmt_csv else OutputFormat.MARKDOWN if fmt_md else OutputFormat.XML
        flat_results =[]
        for r in results:
            flat = {"id": r.get("id"), "score": r.get("score")}
            if "fields" in r:
                flat.update(r["fields"])
            flat_results.append(flat)
            
        console.print(OutputFormatter().format(flat_results, format=fmt))
        return

    # 3. Handle CLI Formatting
    console.print(f"\n[bold]{title}[/bold] [dim](took {elapsed:.4f}s)[/dim]\n")

    if explain_data:
        console.print("[bold magenta]Explanation Data:[/bold magenta]")
        console.print_json(data=explain_data)
        console.print("\n---\n")

    if not results:
        console.print("[yellow]No results found[/yellow]")
        return

    if files_only:
        table = Table(title="Matched Files")
        table.add_column("Score", style="magenta")
        table.add_column("ID", style="cyan")
        table.add_column("Source", style="green")
        table.add_column("Context", style="yellow")
        for r in results:
            table.add_row(
                f"{r.get('score', 0):.4f}", 
                r.get('id', 'unknown'), 
                r['fields'].get('source', ''), 
                r['fields'].get('context', '-')
            )
        console.print(table)
        return

    for i, item in enumerate(results, 1):
        score = item.get('score')
        console.print(f"\n{i}. [cyan]Score: {score:.4f}[/cyan]" if score else f"\n{i}. [cyan]Score: N/A[/cyan]")
        
        doc_id = item.get('id', 'unknown')
        fields = item.get('fields', {})
        source = fields.get('source', 'unknown')
        context = fields.get('context')
        content = fields.get('content', '')

        console.print(f"   [bold]ID:[/bold] {doc_id}")
        console.print(f"   [bold]Source:[/bold] {source}")
        if context:
            console.print(f"   [bold]Context:[/bold] {context}")

        if full:
            console.print(f"\n[bold]Content:[/bold]\n{content}")
        elif line_numbers:
            console.print(f"\n[bold]Content:[/bold]")
            for j, line in enumerate(content.split('\n'), 1):
                console.print(f"   {j:3d}: {line}")
        else:
            preview = content[:150].replace('\n', ' ') + ("..." if len(content) > 150 else "")
            console.print(f"   [bold]Preview:[/bold] {preview}")


# =========================================================================
# CLI Commands
# =========================================================================

@click.group()
@click.version_option(version="0.1.0")
def cli():
    """zrag: Local-first, high-performance CLI application and embedded RAG engine."""
    pass


@cli.command()
@click.argument("query")
@search_options
def search(query: str, collection: Optional[str], top_k: Optional[int], filter_expr: Optional[str], threshold: Optional[float], return_all: bool, full: bool, line_numbers: bool, files_only: bool, fmt_json: bool, fmt_csv: bool, fmt_md: bool, fmt_xml: bool, explain: bool):
    """Perform BM25 keyword search."""
    if return_all: top_k = 1024
    
    with ensure_daemon() as client:
        if not (fmt_json or fmt_csv or fmt_md or fmt_xml):
            console.print(f"[cyan]Searching with BM25: {query}[/cyan]")
            
        res = client.post("/search/bm25", json={"collection_name": collection or "default", "query": query, "top_k": top_k, "filter_expr": filter_expr})
        
        print_search_results(
            res.get("results",[]), res.get("elapsed", 0.0), "BM25 Search Results", None,
            threshold, full, line_numbers, files_only, fmt_json, fmt_csv, fmt_md, fmt_xml
        )


@cli.command()
@click.argument("query")
@search_options
def vsearch(query: str, collection: Optional[str], top_k: Optional[int], filter_expr: Optional[str], threshold: Optional[float], return_all: bool, full: bool, line_numbers: bool, files_only: bool, fmt_json: bool, fmt_csv: bool, fmt_md: bool, fmt_xml: bool, explain: bool):
    """Perform dense vector search."""
    if return_all: top_k = 1024
    
    with ensure_daemon() as client:
        if not (fmt_json or fmt_csv or fmt_md or fmt_xml):
            console.print(f"[cyan]Searching with vectors: {query}[/cyan]")
            
        res = client.post("/search/vector", json={"collection_name": collection or "default", "query": query, "top_k": top_k, "filter_expr": filter_expr, "explain": explain})
        
        print_search_results(
            res.get("results",[]), res.get("elapsed", 0.0), "Vector Search Results", res.get("explain"),
            threshold, full, line_numbers, files_only, fmt_json, fmt_csv, fmt_md, fmt_xml
        )


@cli.command()
@click.argument("query")
@search_options
@click.option("--no-expansion", is_flag=True, help="Disable query expansion")
@click.option("--no-hyde", is_flag=True, help="Disable HyDE")
def query(query: str, collection: Optional[str], top_k: Optional[int], filter_expr: Optional[str], no_expansion: bool, no_hyde: bool, threshold: Optional[float], return_all: bool, full: bool, line_numbers: bool, files_only: bool, fmt_json: bool, fmt_csv: bool, fmt_md: bool, fmt_xml: bool, explain: bool):
    """Perform hybrid search with BM25 + Dense + RRF."""
    if return_all: top_k = 1024
    
    with ensure_daemon() as client:
        if not (fmt_json or fmt_csv or fmt_md or fmt_xml):
            console.print(f"[cyan]Searching with hybrid: {query}[/cyan]")
            
        res = client.post("/search/hybrid", json={
            "collection_name": collection or "default", "query": query, "top_k": top_k,
            "filter_expr": filter_expr, "use_expansion": not no_expansion, "use_hyde": not no_hyde, "explain": explain
        })
        
        print_search_results(
            res.get("results",[]), res.get("elapsed", 0.0), "Hybrid Search Results", res.get("explain"),
            threshold, full, line_numbers, files_only, fmt_json, fmt_csv, fmt_md, fmt_xml
        )


@cli.command()
@click.argument("identifier")
@click.option("-c", "--collection", help="Collection name")
@click.option("--format", "fmt", type=click.Choice(["json", "csv", "xml", "markdown", "cli", "files"]), default="cli", help="Output format")
@click.option("--from", "from_num", type=int, help="Starting offset for pagination")
@click.option("--limit", type=int, help="Maximum number of results to return")
@click.option("--max-bytes", type=int, help="Maximum total bytes to return")
def get(identifier: str, collection: Optional[str], fmt: str, from_num: Optional[int], limit: Optional[int], max_bytes: Optional[int]):
    """Get document(s) by identifier (docid, glob:pattern, or path:line-range)."""
    # Extract collection name from zrag:// URI if not provided
    if not collection and identifier.startswith("zrag://"):
        parts = identifier[7:].split("/", 1)
        if parts:
            collection = parts[0]

    with ensure_daemon() as client:
        if fmt == "cli":
            console.print(f"[cyan]Getting: {identifier}[/cyan]")

        result = client.post("/get", json={
            "collection_name": collection or "default", "identifier": identifier,
            "format": fmt, "from_num": from_num, "limit": limit, "max_bytes": max_bytes,
        })

        if fmt in ("json", "csv", "xml", "markdown"):
            formatter = OutputFormatter()
            out_fmt = OutputFormat.JSON if fmt=="json" else OutputFormat.CSV if fmt=="csv" else OutputFormat.MARKDOWN if fmt=="markdown" else OutputFormat.XML
            if isinstance(result, list):
                flat_results =[]
                for r in result:
                    flat = {"id": r.get("id"), "score": r.get("score")}
                    if "fields" in r:
                        flat.update(r["fields"])
                    flat_results.append(flat)
                console.print(formatter.format(flat_results, format=out_fmt))
            else:
                flat = {"id": result.get("id"), "score": result.get("score")}
                if "fields" in result:
                    flat.update(result["fields"])
                console.print(formatter.format_get_result(flat, format=out_fmt))
            return
            
        if fmt == "files":
            if isinstance(result, list):
                table = Table(title="Files")
                table.add_column("ID", style="cyan")
                table.add_column("Source", style="green")
                for r in result:
                    table.add_row(r.get("id", ""), r.get("fields", {}).get("source", ""))
                console.print(table)
            return

        # CLI Format
        if isinstance(result, list):
            table = Table(title="Results")
            table.add_column("#", style="cyan", width=4)
            table.add_column("Source", style="blue", width=40)
            table.add_column("Preview", style="white", width=60)
            for i, item in enumerate(result, 1):
                source = item.get("fields", {}).get("source", "unknown")
                content = item.get("fields", {}).get("content", "")
                preview = content[:60].replace('\n', ' ') + ("..." if len(content) > 60 else "")
                table.add_row(str(i), source, preview)
            console.print(table)
        else:
            console.print(f"[bold]ID:[/bold] {result.get('id', '')}")
            if result.get("fields", {}).get("line_range"):
                console.print(f"[bold]Line Range:[/bold] {result['fields']['line_range']}")
            console.print(f"[bold]Source:[/bold] {result.get('fields', {}).get('source', 'unknown')}")
            if result.get("fields", {}).get("context"):
                console.print(f"[bold]Context:[/bold] {result['fields']['context']}")
            console.print(f"\n[bold]Content:[/bold]\n{result.get('fields', {}).get('content', '')}")


@cli.command()
def status():
    """Show daemon and collection status."""
    start_time = time.time()
    with ensure_daemon() as client:
        status_data = client.get("/status")
        elapsed = time.time() - start_time
        
        console.print(f"\n[bold]zrag Status[/bold]\n")
        console.print(f"Daemon: [green]Running[/green]")
        console.print(f"Collections loaded: {status_data['collections_loaded']}")
        console.print(f"Models loaded: {'[green]Yes[/green]' if status_data['models_loaded'] else '[yellow]No[/yellow]'}")
        console.print(f"\nResponse time: [cyan]{elapsed:.4f}s[/cyan]")


# =========================================================================
# Collection Groups
# =========================================================================

@cli.group()
def collection():
    """Collection management commands."""
    pass

@collection.command("add")
@click.argument("name")
@click.argument("path", type=click.Path(exists=True), required=False)
@click.option("--mask", help="File mask pattern (e.g., '**/*.md')")
@click.option("--description", help="Collection description")
def collection_add(name: str, path: Optional[str], mask: Optional[str], description: Optional[str]):
    with ensure_daemon() as client:
        if path:
            path_obj = Path(path)
            console.print(f"[cyan]Creating collection '{name}' from {path}...[/cyan]")
            source_path = str(path_obj.absolute())
        else:
            console.print(f"[cyan]Creating empty collection '{name}'...[/cyan]")
            source_path = None

        result = client.post("/collections", json={
            "name": name, "description": description,
            "mask": mask, "source_path": source_path,
        })

        console.print(f"[green]✓ Collection '{name}' created successfully[/green]")
        console.print(f"  Path: {result['path']}\n  Size: {result['size_bytes']} bytes")

@collection.command("list")
def collection_list():
    with ensure_daemon() as client:
        collections = client.get("/collections")
        if not collections:
            console.print("[yellow]No collections found[/yellow]")
            return
        
        table = Table(title="Collections")
        table.add_column("Name", style="cyan")
        table.add_column("Documents", style="magenta")
        table.add_column("Size", style="green")
        table.add_column("Description", style="blue")
        
        for col in collections:
            table.add_row(col["name"], str(col["document_count"]), f"{col['size_bytes']:,} bytes", (col.get("description") or "")[:30] or "-")
        console.print(table)

@collection.command("remove")
@click.argument("name")
@click.option("--force", is_flag=True, help="Force removal without confirmation")
def collection_remove(name: str, force: bool):
    if not force and not click.confirm(f"Are you sure you want to remove collection '{name}'?"):
        return
    with ensure_daemon() as client:
        client.delete(f"/collections/{name}", params={"force": force})
        console.print(f"[green]✓ Collection '{name}' removed successfully[/green]")

@collection.command("rename")
@click.argument("old_name")
@click.argument("new_name")
def collection_rename(old_name: str, new_name: str):
    with ensure_daemon() as client:
        console.print(f"[cyan]Renaming '{old_name}' to '{new_name}'...[/cyan]")
        result = client.post("/collections/rename", params={"old_name": old_name, "new_name": new_name})
        console.print(f"[green]✓ Renamed successfully to {result['name']}[/green]")

@collection.command("ls")
@click.argument("collection")
@click.option("--filter", help="Filter pattern (e.g., '*.py')")
def collection_ls(collection: str, filter: Optional[str]):
    with ensure_daemon() as client:
        files = client.get(f"/collections/{collection}/files", params={"filter": filter})
        if not files:
            console.print("[yellow]No files found[/yellow]")
            return
        table = Table(title=f"Files in '{collection}'")
        table.add_column("Source", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Chunks", style="green")
        for f in files:
            table.add_row(f["source"], f["file_type"], str(f["chunk_count"]))
        console.print(table)

@collection.command("inspect")
@click.argument("name")
def collection_inspect(name: str):
    with ensure_daemon() as client:
        schema = client.get(f"/collections/{name}/inspect")
        console.print(f"\n[bold]Collection: {schema['name']}[/bold]\n\n[cyan]Fields:[/cyan]")
        for f in schema["fields"]: console.print(f"  • {f['name']}: {f['data_type']}")
        console.print("\n[cyan]Vectors:[/cyan]")
        for v in schema["vectors"]:
            console.print(f"  • {v['name']}: {v['data_type']}")
            if v["dimension"]: console.print(f"    Dimension: {v['dimension']}")
            if v["index_type"]: console.print(f"    Index: {v['index_type']}")

@collection.command("optimize")
@click.argument("name")
def collection_optimize(name: str):
    with ensure_daemon() as client:
        console.print(f"[cyan]Optimizing '{name}'...[/cyan]")
        client.post(f"/collections/{name}/optimize")
        console.print(f"[green]✓ '{name}' optimized successfully[/green]")


# =========================================================================
# Orchestration Commands
# =========================================================================

@cli.command()
@click.argument("collection")
@click.option("--pull", is_flag=True, help="Run git pull before scanning")
def update(collection: str, pull: bool):
    """Update a collection by scanning for filesystem changes."""
    with ensure_daemon() as client:
        console.print(f"[cyan]Updating collection '{collection}'...[/cyan]")
        res = client.post(f"/collections/{collection}/update", params={"pull": pull})
        
        console.print(f"[green]✓ Updated successfully[/green] [dim](took {res.get('elapsed', 0):.4f}s)[/dim]")
        console.print(f"  Added: {res['added']} | Updated: {res['updated']} | Removed: {res['removed']}")
        if res.get('errors'):
            console.print(f"[yellow]Errors:[/yellow]")
            for err in res['errors']: console.print(f"  - {err}")

@cli.command()
@click.argument("collection")
@click.option("-f", "--force", is_flag=True, help="Force re-embedding all documents")
def embed(collection: str, force: bool):
    """Generate embeddings for a collection."""
    with ensure_daemon() as client:
        console.print(f"[cyan]Generating embeddings for '{collection}'...[/cyan]")
        res = client.post(f"/collections/{collection}/embed", params={"force": force})
        console.print(f"[green]✓ Embeddings generated[/green] [dim](took {res.get('elapsed', 0):.4f}s)[/dim]")


# =========================================================================
# Ingestion Groups
# =========================================================================

@cli.group()
def ingest():
    """Ingestion commands."""
    pass

@ingest.command("file")
@click.argument("collection_name")
@click.argument("filepath", type=click.Path(exists=True))
@click.option("--source", help="Optional source identifier")
def ingest_file_cmd(collection_name: str, filepath: str, source: Optional[str]):
    with ensure_daemon() as client:
        abs_path = str(Path(filepath).resolve())
        console.print(f"[cyan]Ingesting {abs_path}...[/cyan]")
        res = client.post("/ingest/file", json={"collection_name": collection_name, "filepath": abs_path, "source": source})
        console.print(f"[green]✓ File ingested[/green] [dim](took {res.get('elapsed', 0):.4f}s)[/dim]")

@ingest.command("url")
@click.argument("collection_name")
@click.argument("url")
@click.option("--timeout", default=10, type=int, help="Request timeout in seconds")
def ingest_url_cmd(collection_name: str, url: str, timeout: int):
    with ensure_daemon() as client:
        console.print(f"[cyan]Ingesting URL {url}...[/cyan]")
        res = client.post("/ingest/url", json={"collection_name": collection_name, "url": url, "timeout": timeout})
        console.print(f"[green]✓ URL ingested[/green] [dim](took {res.get('elapsed', 0):.4f}s)[/dim]")


# =========================================================================
# Context Groups
# =========================================================================

@cli.group()
def context():
    """Context management commands."""
    pass

@context.command("add")
@click.argument("path")
@click.argument("description")
def context_add(path: str, description: str):
    with ensure_daemon() as client:
        res = client.post("/context/add", json={"path": path, "description": description})
        console.print(f"[green]✓ Context added to {res['path']}[/green]")

@context.command("remove")
@click.argument("path")
def context_remove(path: str):
    with ensure_daemon() as client:
        # Use params dictionary so httpx properly URL-encodes special characters like *
        res = client.delete("/context/remove", params={"path": path})
        console.print(f"[green]✓ {res.get('message', 'Context removed successfully')}[/green]")

@context.command("list")
@click.option("-c", "--collection", help="Collection name to filter by")
def context_list(collection: Optional[str]):
    with ensure_daemon() as client:
        res = client.get("/context/list", params={"collection_name": collection})
        if not res:
            console.print("[yellow]No contexts found[/yellow]")
            return
        table = Table(title="Contexts")
        table.add_column("Path", style="cyan")
        table.add_column("Description", style="magenta")
        for ctx in res:
            table.add_row(ctx['path'], ctx['description'][:60] or "-")
        console.print(table)

@context.command("check")
@click.argument("collection_name")
def context_check(collection_name: str):
    with ensure_daemon() as client:
        console.print(f"[cyan]Checking context for '{collection_name}'...[/cyan]")
        res = client.get(f"/context/check/{collection_name}")
        missing = res.get('missing_context', [])
        
        if not missing:
            console.print("[green]✓ All sources have context[/green]")
        else:
            console.print(f"[yellow]Found {len(missing)} sources without context:[/yellow]")
            for path in missing[:15]:
                console.print(f"  - {path}")
            if len(missing) > 15:
                console.print(f"  ... and {len(missing) - 15} more")


# =========================================================================
# Daemon Controls
# =========================================================================

@cli.group()
def daemon():
    """Daemon management commands."""
    pass

@daemon.command("start")
@click.option("--host", default=DAEMON_HOST, help="Host to bind to")
@click.option("--port", default=DAEMON_PORT, type=int, help="Port to bind to")
@click.option("--log-level", default="info", help="Log level")
def daemon_start(host: str, port: int, log_level: str):
    """Start the daemon server securely in the background."""
    import os
    import subprocess
    
    # Target condition: If this environment variable is active, we are the background child.
    if os.environ.get("ZRAG_RUN_SERVER") == "1":
        from zrag.daemon import run_daemon
        run_daemon(host=host, port=port, log_level=log_level)
        return

    # If already running, do nothing
    if check_daemon():
        console.print(f"[green]✓ Daemon is already running on {host}:{port}[/green]")
        return

    console.print(f"[cyan]Starting zrag daemon on {host}:{port}...[/cyan]")

    # Pass the environment flag so the child knows to run the server
    env = os.environ.copy()
    env["ZRAG_RUN_SERVER"] = "1"

    # Ensure robust execution regardless of entry point
    if sys.argv[0].endswith('.py'):
        cmd =[sys.executable, sys.argv[0], "daemon", "start", "--host", host, "--port", str(port), "--log-level", log_level]
    else:
        cmd = [sys.argv[0], "daemon", "start", "--host", host, "--port", str(port), "--log-level", log_level]

    # Spawn purely detached process
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True
    )

    # Active polling allows the CLI to give the user back terminal control *immediately* 
    # after models are loaded, rather than keeping them hanging in a detached state.
    with console.status("[cyan]Loading models and starting server...[/cyan]"):
        for _ in range(900):  # Maximum wait time of 60 seconds
            try:
                res = httpx.get(f"http://{host}:{port}/status", timeout=0.1)
                if res.status_code == 200 and res.json().get("models_loaded"):
                    console.print(f"[green]✓ Daemon successfully started and detached on {host}:{port}[/green]")
                    sys.exit(0)
            except httpx.RequestError:
                pass
            
            if process.poll() is not None:
                console.print(f"[red]Error: Daemon process crashed during startup (Exit code: {process.returncode}).[/red]")
                sys.exit(1)
                
            time.sleep(0.1)
            
    console.print("[red]Error: Timeout waiting for daemon to start. Check port availability.[/red]")
    sys.exit(1)


@daemon.command("stop")
def daemon_stop():
    """Stop the daemon gracefully."""
    console.print("[cyan]Stopping zrag daemon...[/cyan]")
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.post(f"{DAEMON_URL}/shutdown")
            response.raise_for_status()
            console.print("[green]✓ Daemon stopped gracefully[/green]")
    except Exception:
        # Fallback: Force kill the background daemon process
        import subprocess
        result = subprocess.run(["pkill", "-f", "daemon start"], capture_output=True)
        if result.returncode == 0:
            console.print("[green]✓ Daemon stopped (process killed)[/green]")
        else:
            console.print("[yellow]Daemon process not found or already stopped.[/yellow]")

# =========================================================================
# MCP Server Controls
# =========================================================================

@cli.command()
def mcp():
    """Run MCP server over stdio (for Claude Desktop, Cursor, etc.).

    For HTTP/browser access (llama.cpp), the MCP endpoint is already built into
    the daemon. Start the daemon first, then connect to http://127.0.0.1:8765/mcp
    """
    from zrag.mcp_server import run_stdio
    run_stdio()


@cli.command()
@click.option("--out", default="SKILL.md", help="Output path for the skill markdown file")
def mcp_skill(out: str):
    """Generate an Agent Skill file (SKILL.md) for Claude Code / MCP integration."""
    content = """---
name: zrag
description: Search and retrieve context from the local zrag RAG system.
---

# zrag

This skill provides semantic and keyword search across the local codebase and documents using the `zrag` RAG system.

## Setup

Start the daemon: `zrag daemon start`
The MCP endpoint is available at `http://127.0.0.1:8765/mcp` (streamable HTTP).

## Tools Available via MCP
- `search(query, collection, top_k, full_content)`: BM25 keyword search.
- `vsearch(query, collection, top_k, full_content)`: Dense vector semantic search.
- `hybrid_query(query, collection, top_k, full_content)`: Hybrid BM25 + Vector search with Reciprocal Rank Fusion.
- `retrieve_lines(identifier, collection, limit)`: Fetch full document or specific lines (e.g., 'path/to/file.py:10-20').
- `list_collections()`: List all available collections.

## Usage Guidelines
1. Always prefer `hybrid_query` for general knowledge retrieval as it provides the most accurate fused results.
2. If looking for exact function names or error codes, use `search` (BM25).
3. When you need to read the full context of a file referenced in a search result, use `retrieve_lines` with the file path.
4. Search results show previews by default (300 chars). Use `full_content=True` to get complete content.
"""
    with open(out, 'w') as f:
        f.write(content)
    console.print(f"[green]✓ Skill file generated at {out}[/green]")


def main():
    cli()

if __name__ == "__main__":
    main()
