"""
zrag: Local-first, high-performance CLI application and embedded RAG engine.

Built on top of zvec vector database, providing end-to-end solution for
ingesting, chunking, embedding, and querying multi-modal knowledge bases.
"""

__version__ = "0.1.0"

from zrag.config import Config
from zrag.core import ZragEngine
from zrag.embeddings import EmbeddingManager
from zrag.sdk import ZragSDK, SearchResult, LineRangeResult, SearchStrategy
from zrag.formatters import OutputFormat, OutputFormatter

__all__ =[
    "Config",
    "ZragEngine",
    "EmbeddingManager",
    "ZragSDK",
    "SearchResult",
    "LineRangeResult",
    "SearchStrategy",
    "OutputFormat",
    "OutputFormatter",
    "__version__",
]
