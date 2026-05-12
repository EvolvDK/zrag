"""
Configuration management for zrag.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


@dataclass
class Config:
    """Configuration for zrag application."""
    
    # Paths
    data_dir: Path = field(default_factory=lambda: Path.home() / ".zrag" / "data")
    collections_dir: Path = field(default_factory=lambda: Path.home() / ".zrag" / "collections")
    
    # Daemon settings
    daemon_host: str = "127.0.0.1"
    daemon_port: int = 8765
    daemon_timeout: int = 300  # 5 minutes idle timeout
    
    # Chunking
    chunk_tokens: int = 900
    chunk_overlap: float = 0.15  # 15%
    
    # Search
    top_k: int = 5
    hnsw_ef: int = 100
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    
    # Model configuration
    dense_embedding_type: str = "local"  # "local", "openai", "qwen", "jina"
    dense_embedding_config: Dict[str, Any] = field(default_factory=dict)
    
    image_embedding_type: str = "clip"  # "clip", "multimodal"
    image_embedding_config: Dict[str, Any] = field(default_factory=dict)
    
    reranker_type: str = "local"  # "local", "qwen", "openai"
    reranker_config: Dict[str, Any] = field(default_factory=dict)
    
    # Query expansion
    use_hyde: bool = True
    query_expansion_api_url: str = "http://localhost:8001/v1"
    query_expansion_model: str = "qmd-query-expansion-1.7B-gguf"
    
    # Reranking
    rerank_candidates: int = 30
    use_reranker_for_bm25: bool = False
    
    # RRF
    rrf_k: int = 30
    
    @property
    def text_dim(self) -> int:
        default_dims = {"local": 384, "openai": 1536, "qwen": 256, "jina": 1024}
        return self.dense_embedding_config.get(
            "dimension",
            default_dims.get(self.dense_embedding_type, 384)
        )
    
    @classmethod
    def from_file(cls, config_path: Path) -> "Config":
        if not config_path.exists():
            return cls()
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f) or {}
        return cls(**config_data)
    
    def to_file(self, config_path: Path) -> None:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
    
    @classmethod
    def load(cls) -> "Config":
        config_path = Path.home() / ".zrag" / "config.yaml"
        return cls.from_file(config_path)
    
    def save(self) -> None:
        config_path = Path.home() / ".zrag" / "config.yaml"
        self.to_file(config_path)
    
    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.collections_dir.mkdir(parents=True, exist_ok=True)
