# zrag

Local-first, high-performance RAG orchestration engine built on zvec. Provides sub-1.0s query latency through a persistent client-daemon architecture with hot-loaded models and indexes.

## Features

- **Persistent Daemon Architecture**: Background daemon keeps zvec collections and embedding models hot in memory
- **Multi-Vector Search**: Combines BM25 sparse vectors with dense semantic vectors via Reciprocal Rank Fusion
- **Smart Chunking**: AST-aware code chunking (Python, TypeScript, Go, Rust, C++) and semantic Markdown splitting
- **Multi-Format Support**: Text, Markdown, code files, PDFs, images, and web URLs
- **Query Expansion**: HyDE-based hypothesis generation with configurable LLM backend
- **Model Context Protocol**: Native MCP server for AI assistant integration (Claude Desktop, Cursor, etc.)
- **Python SDK**: Typed SDK for programmatic access

## Installation

```bash
uv pip install -e .
```

## Quick Start

```bash
# Start the daemon (auto-started on first command)
zrag daemon start

# Create a collection from a directory
zrag collection add ./src --name my-project

# Search using hybrid retrieval
zrag query "authentication middleware"
```

## Configuration

Configuration is stored in `~/.zrag/config.yaml`. Generate a default config:

```python
from zrag.config import Config
config = Config()
config.save()
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `data_dir` | `~/.zrag/data` | Internal data directory |
| `collections_dir` | `~/.zrag/collections` | Collection storage directory |
| `daemon_host` | `127.0.0.1` | Daemon bind address |
| `daemon_port` | `8765` | Daemon HTTP port |
| `daemon_timeout` | `300` | Idle timeout before auto-shutdown (seconds) |
| `chunk_tokens` | `900` | Target chunk size in tokens |
| `chunk_overlap` | `0.15` | Chunk overlap ratio (15%) |
| `top_k` | `5` | Default number of search results |
| `hnsw_ef` | `100` | HNSW search parameter (higher = better recall) |
| `hnsw_m` | `16` | HNSW M parameter (connections per node) |

### Embedding Models

Configure via `dense_embedding_type` and `image_embedding_type`:

| Type | Description | Configuration |
|------|-------------|---------------|
| `local` | Default local embedding (384 dims) | None |
| `openai` | OpenAI API | `api_key`, `model`, `dimension` |
| `qwen` | Dashscope API | `api_key`, `model`, `dimension` |
| `jina` | Jina API | `api_key`, `model`, `dimension`, `task` |
| `clip` | OpenCLIP (images, 512 dims) | None |
| `multimodal` | Generic multimodal model | `model_path`, `dimension`, `torch_dtype` |

### Reranker

Configure via `reranker_type`:

| Type | Description | Configuration |
|------|-------------|---------------|
| `none` | Pure RRF fusion | None |
| `local` | Cross-encoder (ms-marco-MiniLM-L-6-v2) | None |
| `qwen` | Dashscope API | `api_key`, `model` |
| `openai` | OpenAI API | `api_key`, `model` |

### Query Expansion

| Option | Default | Description |
|--------|---------|-------------|
| `use_hyde` | `true` | Enable HyDE hypothesis generation |
| `query_expansion_api_url` | `http://localhost:8001/v1` | Expansion model endpoint |
| `query_expansion_model` | `qmd-query-expansion-1.7B-gguf` | Expansion model name |

## CLI Commands

### Daemon

```bash
zrag daemon start [--host HOST] [--port PORT] [--log-level LEVEL]
zrag daemon stop
zrag status
```

### Collection Management

```bash
# Create collection from directory
zrag collection add <path> --name <name> [--mask "**/*.py"] [--description "..."]

# List all collections
zrag collection list

# Remove collection
zrag collection remove <name> [--force]

# Rename collection
zrag collection rename <old_name> <new_name>

# List files in collection
zrag collection ls <collection> [--filter "*.py"]

# Inspect collection schema
zrag collection inspect <name>

# Optimize collection indexes
zrag collection optimize <name>
```

### Search

```bash
# BM25 keyword search
zrag search "exact term" [-c <collection>] [-n 10]

# Dense vector semantic search
zrag vsearch "semantic query" [-c <collection>] [-n 10]

# Hybrid search (BM25 + Vector + RRF)
zrag query "combined search" [-c <collection>] [-n 10] [--no-expansion] [--no-hyde] [--explain]
```

Search options:
- `-c, --collection`: Collection name (default: "default")
- `-n, --top-k`: Number of results
- `--filter`: zvec filter expression
- `--threshold`: Minimum score threshold
- `--full`: Show full content
- `--line-numbers`: Display content with line numbers
- `--files`: Show only file information
- `--json`, `--csv`, `--md`, `--xml`: Output format
- `--explain`: Show query expansion and reranking details

### Document Retrieval

```bash
# Get by docid, glob pattern, or line range
zrag get doc_abc123 --format json
zrag get "glob:*.py" --limit 50
zrag get "zrag://doc/example.md:10-50"
```

### Collection Update

```bash
# Update collection with filesystem changes
zrag update <collection> [--pull]

# Force re-embed all documents
zrag embed <collection> [--force]
```

### Ingestion

```bash
# Ingest a single file
zrag ingest file <collection> <filepath>

# Ingest content from URL
zrag ingest url <collection> <url> [--timeout 10]
```

### Context Management

```bash
# Add context description to a path
zrag context add "zrag://my-project/legacy" "Legacy v1 API - deprecate soon"

# List all contexts
zrag context list [-c <collection>]

# Remove context
zrag context remove <path>

# Check for missing context annotations
zrag context check <collection>
```

### MCP Server

```bash
# Run MCP server over stdio
zrag mcp

# Generate agent skill file
zrag mcp_skill [--out SKILL.md]
```

The MCP HTTP endpoint is available at `http://127.0.0.1:8765/mcp` when the daemon is running.

## Architecture

```
CLI Client (httpx) --> HTTP --> Daemon (FastAPI/uvicorn)
                                         |
                                         +-- ZragEngine
                                         |       |
                                         |       +-- zvec Collection
                                         |       +-- EmbeddingManager
                                         |       +-- ContextManager
                                         |
                                         +-- MCP Server (stdio/HTTP)
```

The daemon:
1. Initializes zvec and loads embedding models on startup
2. Keeps collections hot in memory for instant queries
3. Auto-shuts down after configurable idle timeout to free memory

## Python SDK

```python
from zrag.sdk import ZragSDK, SearchStrategy

sdk = ZragSDK(engine)

# Hybrid search
results = sdk.search("my-project", "authentication", strategy=SearchStrategy.HYBRID)

# Get document by ID or path
doc = sdk.get("my-project", "src/auth.py:50-80")

# Process results
for result in results:
    print(f"{result.score}: {result.source}")
    print(result.content)
```

## Development

```bash
# Install dependencies
uv pip install -e ".[dev]"

# Run tests
pytest
```

## File Structure

```
zrag/
    cli.py         # CLI commands and client
    daemon.py      # HTTP daemon server
    core.py        # Main engine (collections, search, ingestion)
    embeddings.py  # Embedding model management
    chunking.py    # Text chunking strategies
    file_processing.py  # PDF, URL, image processing
    context.py     # Hierarchical context management
    formatters.py  # Output formatters (JSON, CSV, XML, Markdown)
    config.py      # Configuration management
    sdk.py         # Python SDK
    mcp_server.py  # Model Context Protocol server
```

## Acknowledgments
Thanks to Alibaba for the lighting fast development of [zvec](https://zvec.org/en/), the vector database.
Strongly inspired by [qmd](https://github.com/tobi/qmd), by Tobias Lütke, Shopify CEO.
