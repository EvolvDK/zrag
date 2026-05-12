"""
HTTP daemon server for zrag.

Provides a persistent daemon that keeps zvec collections and embedding models
hot in memory for zero-latency CLI operations.
"""

import os
import sys
import asyncio
import signal
import time
import threading
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from zrag.config import Config
from zrag.core import ZragEngine

# Global state
engine: Optional[ZragEngine] = None
shutdown_event = asyncio.Event()
server: Optional[uvicorn.Server] = None
last_accessed = time.time()


# =============================================================================
# Pydantic Request & Response Models
# =============================================================================

class CollectionCreateRequest(BaseModel):
    name: str = Field(..., description="Collection name")
    description: Optional[str] = Field(None, description="Collection description")
    mask: Optional[str] = Field(None, description="File mask pattern")
    source_path: Optional[str] = Field(None, description="Source directory path to ingest from")

class IngestTextRequest(BaseModel):
    collection_name: str = Field(..., description="Collection name")
    text: str = Field(..., description="Text content to ingest")
    source: str = Field(..., description="Source identifier")
    file_type: str = Field("text", description="File type")

class IngestFileRequest(BaseModel):
    collection_name: str = Field(..., description="Collection name")
    filepath: str = Field(..., description="Path to the file")
    source: Optional[str] = Field(None, description="Optional source identifier")

class IngestUrlRequest(BaseModel):
    collection_name: str = Field(..., description="Collection name")
    url: str = Field(..., description="URL to fetch content from")
    timeout: int = Field(10, description="Request timeout in seconds")

class SearchRequest(BaseModel):
    collection_name: str
    query: str
    top_k: Optional[int] = None
    filter_expr: Optional[str] = None
    explain: bool = False

class HybridSearchRequest(BaseModel):
    collection_name: str
    query: str
    top_k: Optional[int] = None
    filter_expr: Optional[str] = None
    use_expansion: bool = True
    use_hyde: bool = True
    explain: bool = False
    
class ContextAddRequest(BaseModel):
    path: str = Field(..., description="Path (physical, virtual, or 'global')")
    description: str = Field(..., description="Human-readable description")

class GetRequest(BaseModel):
    collection_name: str
    identifier: str
    format: str = "cli"
    from_num: Optional[int] = None
    limit: Optional[int] = None
    max_bytes: Optional[int] = None
    
class StatusResponse(BaseModel):
    daemon_running: bool
    collections_loaded: int
    models_loaded: bool


# =============================================================================
# Background Tasks & Lifecycle
# =============================================================================

async def idle_timeout_task():
    config = Config()
    timeout = config.daemon_timeout
    while not shutdown_event.is_set():
        await asyncio.sleep(10)
        if time.time() - last_accessed > timeout:
            print(f"\nDaemon idle for {timeout}s. Shutting down gracefully to free memory.")
            shutdown_event.set()
            if server: server.should_exit = True
            break

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    print("Starting zrag daemon...")
    config = Config()
    config.ensure_directories()
    engine = ZragEngine(config)
    engine.preload_resources()
    print(f"✓ Daemon hot and ready on {config.daemon_host}:{config.daemon_port}")

    # Start MCP HTTP server in background (auto-stopped on shutdown)
    from zrag.mcp_server import run_http
    mcp_thread = threading.Thread(target=run_http, args=(8002,), daemon=True)
    mcp_thread.start()
    print(f"✓ MCP endpoint proxied at http://{config.daemon_host}:{config.daemon_port}/mcp")

    timeout_task = asyncio.create_task(idle_timeout_task())
    yield
    timeout_task.cancel()
    print("Shutting down zrag daemon...")
    if engine:
        for name in list(engine._collections.keys()):
            engine._collections[name].flush()

app = FastAPI(title="zrag Daemon", lifespan=lifespan)

@app.middleware("http")
async def update_last_accessed(request: Request, call_next):
    global last_accessed
    last_accessed = time.time()
    return await call_next(request)

# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/shutdown")
async def shutdown():
    global shutdown_event, server
    shutdown_event.set()
    if server:
        server.should_exit = True
    return {"message": "Shutting down"}

@app.get("/status", response_model=StatusResponse)
async def get_status():
    if not engine: raise HTTPException(status_code=503, detail="Daemon not initialized")
    return StatusResponse(**engine.get_status())

# --- Collection Management ---

@app.post("/collections")
def create_collection(request: CollectionCreateRequest):
    if not engine: raise HTTPException(status_code=503, detail="Daemon not initialized")
    try:
        info = engine.create_collection(
            name=request.name,
            description=request.description,
            mask=request.mask,
            source_path=request.source_path,
        )
        return {"name": info.name, "path": str(info.path), "document_count": info.document_count, "size_bytes": info.size_bytes}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections")
def list_collections():
    if not engine: raise HTTPException(status_code=503, detail="Daemon not initialized")
    try:
        cols = engine.list_collections()
        return[{"name": c.name, "path": str(c.path), "document_count": c.document_count, "size_bytes": c.size_bytes, "description": c.description} for c in cols]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/collections/{name}")
def remove_collection(name: str, force: bool = False):
    if not engine: raise HTTPException(status_code=503, detail="Daemon not initialized")
    try:
        engine.remove_collection(name, force=force)
        return {"message": f"Collection '{name}' removed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/collections/rename")
def rename_collection(old_name: str = Query(..., description="Old collection name"), new_name: str = Query(..., description="New collection name")):
    if not engine: raise HTTPException(status_code=503, detail="Daemon not initialized")
    try:
        info = engine.rename_collection(old_name, new_name)
        return {"name": info.name, "path": str(info.path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections/{name}/files")
def list_files(name: str, filter: Optional[str] = None):
    if not engine: raise HTTPException(status_code=503, detail="Daemon not initialized")
    try:
        return engine.list_files_in_collection(name, filter)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/collections/{name}/update")
def update_collection(name: str, pull: bool = False):
    if not engine: raise HTTPException(status_code=503, detail="Daemon not initialized")
    try:
        stats, elapsed = engine.update_collection(name, pull)
        return {"added": stats.added, "updated": stats.updated, "unchanged": stats.unchanged, "removed": stats.removed, "errors": stats.errors, "elapsed": elapsed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/collections/{name}/embed")
def embed_collection(name: str, force: bool = False):
    if not engine: raise HTTPException(status_code=503, detail="Daemon not initialized")
    try:
        stats, elapsed = engine.embed_collection(name, force)
        return {"added": stats.added, "updated": stats.updated, "unchanged": stats.unchanged, "removed": stats.removed, "errors": stats.errors, "elapsed": elapsed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections/{name}/inspect")
def inspect_collection(name: str):
    if not engine: raise HTTPException(status_code=503, detail="Daemon not initialized")
    try:
        return engine.inspect_collection(name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/collections/{name}/optimize")
def optimize_collection(name: str):
    if not engine: raise HTTPException(status_code=503, detail="Daemon not initialized")
    try:
        engine.optimize_collection(name)
        return {"message": "Optimization complete"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Ingestion ---

@app.post("/ingest/text")
def ingest_text(request: IngestTextRequest):
    if not engine: raise HTTPException(status_code=503, detail="Daemon not initialized")
    try:
        stats, elapsed = engine.ingest_text(request.collection_name, request.text, request.source, request.file_type)
        return {"added": stats.added, "updated": stats.updated, "unchanged": stats.unchanged, "errors": stats.errors, "elapsed": elapsed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/file")
def ingest_file(request: IngestFileRequest):
    if not engine: raise HTTPException(status_code=503, detail="Daemon not initialized")
    try:
        stats, elapsed = engine.ingest_file(request.collection_name, Path(request.filepath), request.source)
        return {"added": stats.added, "updated": stats.updated, "unchanged": stats.unchanged, "errors": stats.errors, "elapsed": elapsed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/url")
def ingest_url(request: IngestUrlRequest):
    if not engine: raise HTTPException(status_code=503, detail="Daemon not initialized")
    try:
        stats, elapsed = engine.ingest_url(request.collection_name, request.url, request.timeout)
        return {"added": stats.added, "updated": stats.updated, "unchanged": stats.unchanged, "errors": stats.errors, "elapsed": elapsed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Search ---

@app.post("/search/bm25")
async def search_bm25(request: SearchRequest):
    if not engine: raise HTTPException(status_code=503, detail="Daemon not initialized")
    try:
        results, elapsed = await asyncio.to_thread(
            engine.search_bm25, request.collection_name, request.query, request.top_k, request.filter_expr
        )
        return {
            "results":[{"id": r.id, "score": getattr(r, 'score', 0.0), "fields": r.fields, "context": r.fields.get("context", "")} for r in results],
            "elapsed": elapsed,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/vector")
async def search_vector(request: SearchRequest):
    if not engine: raise HTTPException(status_code=503, detail="Daemon not initialized")
    try:
        results, elapsed, explain = await asyncio.to_thread(
            engine.search_vector, request.collection_name, request.query, request.top_k, request.filter_expr, explain=request.explain
        )
        return {
            "results":[{"id": r.id, "score": getattr(r, 'score', 0.0), "fields": r.fields, "context": r.fields.get("context", "")} for r in results],
            "elapsed": elapsed,
            "explain": explain,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/hybrid")
async def search_hybrid(request: HybridSearchRequest):
    if not engine: raise HTTPException(status_code=503, detail="Daemon not initialized")
    try:
        results, elapsed, explain = await asyncio.to_thread(
            engine.search_hybrid,
            request.collection_name,
            request.query,
            request.top_k,
            request.filter_expr,
            request.use_expansion,
            request.use_hyde,
            request.explain,
        )
        return {
            "results":[{"id": r.id, "score": getattr(r, 'score', 0.0), "fields": r.fields, "context": r.fields.get("context", "")} for r in results],
            "elapsed": elapsed,
            "explain": explain,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Context ---

@app.post("/context/add")
def add_context(request: ContextAddRequest):
    if not engine: raise HTTPException(status_code=503, detail="Daemon not initialized")
    try:
        context = engine.add_context(request.path, request.description)
        return {"path": context['path'], "description": context['description'], "created_at": context['created_at'], "updated_at": context['updated_at']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/context/remove")
def remove_context(path: str):
    if not engine: raise HTTPException(status_code=503, detail="Daemon not initialized")
    try:
        count = engine.remove_context(path)
        if count > 0:
            return {"message": f"Removed {count} context(s)", "count": count}
        raise HTTPException(status_code=404, detail="No matching context found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/context/list")
def list_contexts(collection_name: Optional[str] = None):
    if not engine: raise HTTPException(status_code=503, detail="Daemon not initialized")
    try:
        return engine.list_contexts(collection_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/context/check/{collection_name}")
def check_missing_context(collection_name: str):
    if not engine: raise HTTPException(status_code=503, detail="Daemon not initialized")
    try:
        missing = engine.check_missing_context(collection_name)
        return {"collection_name": collection_name, "missing_context": missing}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Get Documents ---

@app.post("/get")
def get_document(request: GetRequest):
    if not engine: raise HTTPException(status_code=503, detail="Daemon not initialized")
    try:
        from zrag.sdk import ZragSDK
        sdk = ZragSDK(engine)
        
        limit = request.limit or 10
        from_num = request.from_num or 0
        top_k = from_num + limit 
        fetch_k = top_k if top_k > 0 else 1024 # Restrict memory footprint natively
        
        result = sdk.get(request.collection_name, request.identifier, top_k=fetch_k)
        
        if isinstance(result, list):
            if request.from_num is not None:
                result = result[request.from_num:]
            if request.limit is not None:
                result = result[:request.limit]
            
            if request.max_bytes is not None:
                total_bytes = 0
                filtered =[]
                for r in result:
                    size = len(r.content.encode('utf-8'))
                    if total_bytes + size <= request.max_bytes:
                        filtered.append(r)
                        total_bytes += size
                    else:
                        if not filtered:
                            r.content = r.content.encode('utf-8')[:request.max_bytes].decode('utf-8', errors='ignore')
                            filtered.append(r)
                        break
                result = filtered
            
            return[{"id": r.id, "score": getattr(r, 'score', None), "fields": {"content": r.content, "source": r.source, "context": getattr(r, 'context', None)}} for r in result]
        
        elif result:
            if request.from_num is not None or request.limit is not None:
                lines = result.content.split('\n')
                start = max(0, (request.from_num or 1) - 1)
                end = start + (request.limit or len(lines))
                result.content = '\n'.join(lines[start:end])
                result.line_range = f"{start+1}-{end}"
                
            if request.max_bytes is not None:
                encoded = result.content.encode('utf-8')
                if len(encoded) > request.max_bytes:
                    result.content = encoded[:request.max_bytes].decode('utf-8', errors='ignore')
 
            return {
                "id": result.id, 
                "fields": {
                    "content": result.content, 
                    "source": result.source, 
                    "context": getattr(result, 'context', None), 
                    "line_range": getattr(result, 'line_range', None)
                }
            }
        
        raise HTTPException(status_code=404, detail="Document not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# MCP Proxy (forwards to internal MCP HTTP server on port 8002)
# =============================================================================

MCP_PORT = 8002

@app.api_route("/mcp/{rest:path}", methods=["GET", "POST", "DELETE", "OPTIONS"])
@app.api_route("/mcp", methods=["GET", "POST", "DELETE", "OPTIONS"])
async def mcp_proxy(request: Request, rest: str = ""):
    """Transparently proxy /mcp requests to the internal MCP HTTP server."""
    mcp_url = f"http://127.0.0.1:{MCP_PORT}/mcp"
    if rest:
        mcp_url += f"/{rest}"
    if request.url.query:
        mcp_url += f"?{request.url.query}"

    body = await request.body()
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.request(
            method=request.method,
            url=mcp_url,
            headers=headers,
            content=body,
        )
        return StreamingResponse(
            resp.aiter_bytes(),
            status_code=resp.status_code,
            headers={
                k: v for k, v in resp.headers.items()
                if k.lower() not in ("content-encoding", "transfer-encoding")
            },
        )


# =============================================================================
# Process Daemonization & Execution
# =============================================================================

def detach_process():
    """Double-fork to completely detach from terminal (UNIX only)."""
    if os.name == 'nt':
        print("Warning: Background daemonize is not natively supported on Windows. Running in foreground.")
        return

    try:
        # First fork
        pid = os.fork()
        if pid > 0:
            # Exit first parent
            sys.exit(0)
    except OSError as e:
        sys.stderr.write(f"Fork #1 failed: {e}\n")
        sys.exit(1)

    # Decouple from parent environment
    os.chdir('/')
    os.setsid()
    os.umask(0)

    try:
        # Second fork
        pid = os.fork()
        if pid > 0:
            # Exit second parent
            sys.exit(0)
    except OSError as e:
        sys.stderr.write(f"Fork #2 failed: {e}\n")
        sys.exit(1)

    # Redirect standard file descriptors to /dev/null
    sys.stdout.flush()
    sys.stderr.flush()
    with open(os.devnull, 'r') as f:
        os.dup2(f.fileno(), sys.stdin.fileno())
    with open(os.devnull, 'a+') as f:
        os.dup2(f.fileno(), sys.stdout.fileno())
        os.dup2(f.fileno(), sys.stderr.fileno())

def run_daemon(host: str = "127.0.0.1", port: int = 8765, log_level: str = "info", daemonize: bool = False):
    """Run the uvicorn server, optionally detaching as a true background daemon."""
    
    if daemonize:
        print(f"Detaching zrag daemon to background... (Host: {host}:{port})")
        detach_process()

    global server
    config = uvicorn.Config(app=app, host=host, port=port, log_level=log_level, access_log=False)
    server = uvicorn.Server(config)
    
    def handle_shutdown(signum, frame):
        shutdown_event.set()
        server.should_exit = True
        
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    server.run()
