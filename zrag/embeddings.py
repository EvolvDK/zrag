"""
Embedding management for zrag.
"""

import os
from pathlib import Path
from typing import List, Dict, Any
import math
import torch

from zvec import Doc
from zvec.extension.rerank_function import RerankFunction
from zvec.extension import (
    DefaultLocalDenseEmbedding,
    OpenAIDenseEmbedding,
    QwenDenseEmbedding,
    JinaDenseEmbedding,
    BM25EmbeddingFunction,
)
from zrag.config import Config


class CustomCrossEncoderReranker(RerankFunction):
            """Integrates local Sentence-Transformers natively into zvec's query pipeline
            using custom position-aware blending between RRF and Cross-Encoder scores."""
            def __init__(self, query: str, topn: int, model, rrf_k: int = 60):
                super().__init__(topn=topn)
                self.query = query
                self.model = model
                self.rrf_k = rrf_k
        
            def rerank(self, query_results: dict[str, list[Doc]]) -> list[Doc]:
                unique_docs = {}
                rrf_scores = {}
                
                # 1. RRF Score Computation
                for doc_list in query_results.values():
                    for rank_0, doc in enumerate(doc_list):
                        if doc.id not in unique_docs:
                            unique_docs[doc.id] = doc
                            rrf_scores[doc.id] = 0.0
                        # Add reciprocal rank for this specific execution branch
                        rrf_scores[doc.id] += 1.0 / (self.rrf_k + rank_0 + 1)
                
                doc_ids = list(unique_docs.keys())
                if not doc_ids:
                    return[]
                
                # Sort by RRF to establish positional rank
                doc_ids.sort(key=lambda did: rrf_scores[did], reverse=True)
                max_rrf = rrf_scores[doc_ids[0]] if doc_ids else 1.0
                
                # 2. Cross-Encoder ML Score Computation
                texts = [unique_docs[did].fields.get("content", "") for did in doc_ids]
                ml_scores = {}
                try:
                    scores = self.model.predict([[self.query, t] for t in texts])
                    for i, did in enumerate(doc_ids):
                        # Sigmoid to normalize unbounded scores to (0, 1)
                        ml_scores[did] = float(1 / (1 + math.exp(-scores[i])))
                except Exception as e:
                    print(f"⚠ Rerank failed: {e}")
                    for did in doc_ids:
                        ml_scores[did] = 0.0
        
                # 3. Position-aware Blending
                final_docs =[]
                for i, did in enumerate(doc_ids):
                    pos = i + 1
                    # Scale RRF to [0,1] so percentages aren't crushed by tiny RRF magnitudes
                    norm_rrf = rrf_scores[did] / max_rrf 
                    rs = ml_scores[did]
                    
                    if pos <= 3:
                        final_score = 0.75 * norm_rrf + 0.25 * rs
                    elif pos <= 10:
                        final_score = 0.60 * norm_rrf + 0.40 * rs
                    else:
                        final_score = 0.40 * norm_rrf + 0.60 * rs
                        
                    orig_doc = unique_docs[did]
                    
                    # Store debug metrics securely inside the mutable fields dictionary
                    fields = orig_doc.fields.copy()
                    fields["_rrf_score"] = rrf_scores[did]
                    fields["_rerank_score"] = rs
                    
                    # Instantiate a fresh Doc to respect zvec C++ bindings
                    final_docs.append(Doc(id=orig_doc.id, score=final_score, fields=fields))
                        
                # Final sort returning exact Top N
                final_docs.sort(key=lambda x: getattr(x, 'score', 0.0), reverse=True)
                return final_docs[:self._topn]

class EmbeddingManager:
    """Manages embedding models with lazy loading."""
    
    def __init__(self, config: Config):
        self.config = config
        self._dense_embed = None
        self._sparse_embed_query = None
        self._sparse_embed_doc = None
        self._clip_model = None
        self._clip_preprocess = None
        
    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    @property
    def dense_embed(self):
        if self._dense_embed is None:
            t = self.config.dense_embedding_type
            cfg = self.config.dense_embedding_config
            dim = self.config.text_dim
            
            if t == "local":
                self._dense_embed = DefaultLocalDenseEmbedding()
            elif t == "openai":
                self._dense_embed = OpenAIDenseEmbedding(
                    api_key=cfg.get("api_key", os.getenv("OPENAI_API_KEY")),
                    model=cfg.get("model", "text-embedding-3-small"),
                    dimension=dim,
                )
            elif t == "qwen":
                self._dense_embed = QwenDenseEmbedding(
                    api_key=cfg.get("api_key", os.getenv("DASHSCOPE_API_KEY")),
                    model=cfg.get("model", "text-embedding-v4"),
                    dimension=dim,
                )
            elif t == "jina":
                self._dense_embed = JinaDenseEmbedding(
                    api_key=cfg.get("api_key", os.getenv("JINA_API_KEY")),
                    model=cfg.get("model", "jina-embeddings-v5-text-small"),
                    dimension=dim,
                    task=cfg.get("task", "text-matching"),
                )
            else:
                raise ValueError(f"Unknown dense embedding type: {t}")
            print(f"✓ Dense embedding ({t}): {dim} dimensions")
        return self._dense_embed
    
    @property
    def sparse_embed_query(self):
        if self._sparse_embed_query is None:
            self._sparse_embed_query = BM25EmbeddingFunction(language="en", encoding_type="query")
        return self._sparse_embed_query
    
    @property
    def sparse_embed_doc(self):
        if self._sparse_embed_doc is None:
            self._sparse_embed_doc = BM25EmbeddingFunction(language="en", encoding_type="document")
        return self._sparse_embed_doc
    
    def embed_dense(self, text: str) -> List[float]:
        try:
            return self.dense_embed.embed(text)
        except Exception as e:
            print(f"⚠ Failed to embed text (dense): {e}")
            return[0.0] * self.config.text_dim

    def embed_sparse_query(self, text: str) -> Dict[int, float]:
        try:
            return self.sparse_embed_query.embed(text)
        except Exception as e:
            print(f"⚠ Failed to embed query (sparse): {e}")
            return {}

    def embed_sparse_doc(self, text: str) -> Dict[int, float]:
        try:
            return self.sparse_embed_doc.embed(text)
        except Exception as e:
            print(f"⚠ Failed to embed document (sparse): {e}")
            return {}
            
    def get_reranker(self, query: str, top_n: int):
        from zvec.extension.multi_vector_reranker import RrfReRanker
        
        if self.config.reranker_type == "none":
            return RrfReRanker(topn=top_n, rank_constant=self.config.rrf_k)
        
        elif self.config.reranker_type == "local":
            if not hasattr(self, "_reranker_model"):
                try:
                    from sentence_transformers import CrossEncoder
                    self._reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=self.device)
                except Exception as e:
                    print(f"⚠ Failed to load local reranker: {e}")
                    self.config.reranker_type = "none"
                    return RrfReRanker(topn=top_n, rank_constant=self.config.rrf_k)
            return CustomCrossEncoderReranker(
                query=query, 
                topn=top_n, 
                model=self._reranker_model, 
                rrf_k=self.config.rrf_k
            )
            
        return RrfReRanker(topn=top_n, rank_constant=self.config.rrf_k)

    def preload_models(self) -> None:
        _ = self.dense_embed
        _ = self.sparse_embed_query
        _ = self.sparse_embed_doc
        print("✓ All embedding models preloaded")
    
    def _load_clip(self):
        if self._clip_model is None:
            try:
                import open_clip
                from PIL import Image
                self._clip_model, _, self._clip_preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-32', pretrained='laion2b_s34b_b79k', device=self.device
                )
                print(f"✓ OpenCLIP model loaded on {self.device}")
            except ImportError:
                print("⚠ open-clip-torch not installed, image embeddings disabled")
                self._clip_model = None
            except Exception as e:
                print(f"⚠ Failed to load OpenCLIP model: {e}")
                self._clip_model = None
    
    def get_image_embedding(self, filepath: Path) -> List[float]:
        if self._clip_model is None:
            self._load_clip()
        if self._clip_model is None:
            return [0.0] * 512
        try:
            from PIL import Image
            image = self._clip_preprocess(Image.open(filepath)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                return self._clip_model.encode_image(image).cpu().numpy()[0].tolist()
        except Exception as e:
            print(f"⚠ Failed to embed image {filepath.name}: {e}")
            return [0.0] * 512

