"""
Core engine for zrag.

Handles collection management, document operations, and zvec integration.
"""

import json
import time
import shutil
import hashlib
import subprocess
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field

import zvec
from zvec import Doc

from zrag.config import Config
from zrag.embeddings import EmbeddingManager
from zrag.chunking import (
    chunk_text_simple,
    chunk_code_ast,
    chunk_text_semantic,
    get_file_type,
)
from zrag.file_processing import (
    extract_text_from_file,
    extract_text_from_url,
)
from zrag.context import ContextManager


@dataclass
class CollectionInfo:
    """Information about a collection."""
    name: str
    path: Path
    document_count: int
    size_bytes: int
    created_at: float
    updated_at: float
    description: Optional[str] = None
    mask: Optional[str] = None
    source_path: Optional[str] = None


@dataclass
class IngestStats:
    """Statistics from ingestion operation."""
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    errors: List[str] = field(default_factory=list)


class ZragEngine:
    """Main engine for zrag, managing collections and embeddings."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.config.ensure_directories()

        self.embeddings = EmbeddingManager(self.config)
        self._collections: Dict[str, zvec.Collection] = {}
        self.context_manager = ContextManager(self.config.data_dir)

        self._zvec_initialized = False

    def _ensure_zvec_initialized(self):
        """Ensure zvec is initialized (lazy initialization)."""
        if not self._zvec_initialized:
            zvec.init(
                log_type=zvec.LogType.CONSOLE,
                log_level=zvec.LogLevel.WARN,
                query_threads=4,
                memory_limit_mb=2048,
            )
            self._zvec_initialized = True
    
    def _get_collection_path(self, name: str) -> Path:
        """Get the filesystem path for a collection."""
        return self.config.collections_dir / name
    
    def _open_collection(self, name: str) -> zvec.Collection:
        """Open a collection, loading it if not already in memory."""
        self._ensure_zvec_initialized()
        if name not in self._collections:
            path = self._get_collection_path(name)
            if not path.exists():
                raise ValueError(f"Collection '{name}' does not exist")

            option = zvec.CollectionOption(
                read_only=False,
                enable_mmap=True,
            )
            self._collections[name] = zvec.open(str(path), option=option)

        return self._collections[name]

    def _inject_context(self, docs: List[Doc]) -> None:
        """Helper to dynamically attach the resolved Context tree to retrieved documents."""
        for doc in docs:
            source = doc.fields.get("source")
            if source:
                contexts = self.context_manager.resolve_context(source)
                if contexts:
                    doc.fields["context"] = " | ".join(contexts)

    # =========================================================================
    # Collection Management
    # =========================================================================

    def create_collection(
        self,
        name: str,
        description: Optional[str] = None,
        mask: Optional[str] = None,
        source_path: Optional[str] = None,
    ) -> CollectionInfo:
        self._ensure_zvec_initialized()
        path = self._get_collection_path(name)

        if path.exists():
            raise ValueError(f"Collection '{name}' already exists")
        
        # Define Schema
        schema = zvec.CollectionSchema(
            name=name,
            fields=[
                zvec.FieldSchema(name="content", data_type=zvec.DataType.STRING),
                zvec.FieldSchema(name="source", data_type=zvec.DataType.STRING),
                zvec.FieldSchema(name="chunk_id", data_type=zvec.DataType.INT32),
                zvec.FieldSchema(name="file_type", data_type=zvec.DataType.STRING),
            ],
            vectors=[
                zvec.VectorSchema(
                    name="text_embedding",
                    data_type=zvec.DataType.VECTOR_FP32,
                    dimension=self.config.text_dim,
                    index_param=zvec.HnswIndexParam(
                        metric_type=zvec.MetricType.COSINE,
                        m=self.config.hnsw_m,
                        ef_construction=self.config.hnsw_ef_construction,
                    ),
                ),
                zvec.VectorSchema(
                    name="image_embedding",
                    data_type=zvec.DataType.VECTOR_FP32,
                    dimension=512,
                    index_param=zvec.HnswIndexParam(
                        metric_type=zvec.MetricType.COSINE,
                        m=self.config.hnsw_m,
                        ef_construction=self.config.hnsw_ef_construction,
                    ),
                ),
                zvec.VectorSchema(
                    name="sparse_embedding",
                    data_type=zvec.DataType.SPARSE_VECTOR_FP32,
                ),
            ],
        )
        
        self._collections[name] = zvec.create_and_open(str(path), schema)

        now = time.time()
        metadata = {
            "mask": mask or "",
            "source_path": source_path or "",
            "created_at": now,
            "updated_at": now,
        }

        with open(path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        if description:
            self.context_manager.add_context(f"zrag://{name}", description)

        if source_path:
            self.update_collection(name, pull=False)

        return self.list_collections_by_name(name)

    def list_collections(self) -> List[CollectionInfo]:
        collections =[]
        for path in self.config.collections_dir.iterdir():
            if not path.is_dir():
                continue
                
            name = path.name
            try:
                collection = self._open_collection(name)
                
                metadata_file = path / "metadata.json"
                metadata = {}
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)

                size_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                document_count = getattr(collection.stats, 'doc_count', 0)
                    
                ctx_node = self.context_manager.get_context(f"zrag://{name}")
                collection_desc = ctx_node.description if ctx_node else None

                collections.append(CollectionInfo(
                    name=name,
                    path=path,
                    document_count=document_count,
                    size_bytes=size_bytes,
                    created_at=metadata.get("created_at", 0.0),
                    updated_at=metadata.get("updated_at", 0.0),
                    description=collection_desc,
                    mask=metadata.get("mask"),
                    source_path=metadata.get("source_path"),
                ))
            except Exception as e:
                print(f"Warning: Could not read collection '{name}': {e}")

        return collections

    def list_collections_by_name(self, name: str) -> CollectionInfo:
        for collection in self.list_collections():
            if collection.name == name:
                return collection
        raise ValueError(f"Collection '{name}' not found")

    def remove_collection(self, name: str, force: bool = False) -> bool:
        path = self._get_collection_path(name)
        if not path.exists():
            raise ValueError(f"Collection '{name}' does not exist")

        if name in self._collections:
            try:
                self._collections[name].destroy()
            except Exception as e:
                if not force: raise
                print(f"⚠ Failed to destroy collection '{name}': {e}")
            del self._collections[name]
            import gc; gc.collect()
        else:
            try:
                collection = zvec.open(str(path), zvec.CollectionOption(read_only=False))
                collection.destroy()
            except Exception as e:
                if not force: raise
                print(f"⚠ Failed to destroy collection '{name}': {e}")
                shutil.rmtree(path, ignore_errors=True)
        return True

    def rename_collection(self, old_name: str, new_name: str) -> CollectionInfo:
        old_path = self._get_collection_path(old_name)
        new_path = self._get_collection_path(new_name)

        if not old_path.exists(): raise ValueError(f"Collection '{old_name}' does not exist")
        if new_path.exists(): raise ValueError(f"Collection '{new_name}' already exists")

        if old_name in self._collections:
            del self._collections[old_name]

        old_path.rename(new_path)

        meta_file = new_path / "metadata.json"
        if meta_file.exists():
            with open(meta_file, 'r') as f: meta = json.load(f)
            meta["updated_at"] = time.time()
            with open(meta_file, 'w') as f: json.dump(meta, f)

        return self.list_collections_by_name(new_name)

    def inspect_collection(self, name: str) -> Dict[str, Any]:
        collection = self._open_collection(name)
        schema = collection.schema
        return {
            "name": schema.name,
            "fields":[{"name": f.name, "data_type": str(f.data_type)} for f in schema.fields],
            "vectors":[
                {
                    "name": v.name,
                    "data_type": str(v.data_type),
                    "dimension": getattr(v, "dimension", None),
                    "index_type": str(type(v.index_param).__name__) if v.index_param else None,
                }
                for v in schema.vectors
            ],
        }

    def optimize_collection(self, name: str) -> None:
        self._open_collection(name).optimize()


    # =========================================================================
    # Orchestration & Updates
    # =========================================================================

    def _get_files_to_ingest(self, source_path: Path, mask: Optional[str]) -> List[Path]:
        if not source_path or not source_path.exists():
            return[]
        if mask:
            return list(source_path.glob(mask))
        extensions =["*.md", "*.txt", "*.py", "*.js", "*.ts", "*.go", "*.rs", "*.cpp", "*.hpp", "*.pdf", "*.png", "*.jpg", "*.jpeg"]
        files =[]
        for ext in extensions:
            files.extend(source_path.rglob(ext))
        return files

    def update_collection(self, collection_name: str, pull: bool = False, force: bool = False) -> Tuple[IngestStats, float]:
        start_time = time.time()
        info = self.list_collections_by_name(collection_name)
        
        if not info.source_path:
            return IngestStats(errors=["Collection has no source_path defined."]), 0.0

        source_path = Path(info.source_path)

        if pull and source_path.is_dir():
            try:
                subprocess.run(["git", "pull"], cwd=source_path, capture_output=True, check=True)
            except subprocess.CalledProcessError as e:
                return IngestStats(errors=[f"Git pull failed: {e.stderr.decode()}"]), time.time() - start_time
            except FileNotFoundError:
                pass 

        files_to_ingest = self._get_files_to_ingest(source_path, info.mask)
        stats = IngestStats()
        
        for filepath in files_to_ingest:
            if filepath.is_file():
                try:
                    f_stats, _ = self.ingest_file(collection_name, filepath, force=force)
                    stats.added += f_stats.added
                    stats.updated += f_stats.updated
                    stats.unchanged += f_stats.unchanged
                    stats.errors.extend(f_stats.errors)
                except Exception as e:
                    stats.errors.append(f"Failed to ingest {filepath}: {e}")

        meta_file = self._get_collection_path(collection_name) / "metadata.json"
        if meta_file.exists():
            with open(meta_file, 'r') as f: meta = json.load(f)
            meta["updated_at"] = time.time()
            with open(meta_file, 'w') as f: json.dump(meta, f)

        try:
            self.optimize_collection(collection_name)
        except Exception:
            pass

        return stats, time.time() - start_time

    def embed_collection(self, collection_name: str, force: bool = False) -> Tuple[IngestStats, float]:
        return self.update_collection(collection_name, pull=False, force=force)

    # =========================================================================
    # Ingestion Pipeline
    # =========================================================================

    def _generate_doc_id(self, source: str, chunk_id: int) -> str:
        source_hash = hashlib.md5(str(source).encode('utf-8')).hexdigest()[:16]
        return f"doc_{source_hash}_{chunk_id}"

    def ingest_text(self, collection_name: str, text: str, source: str, file_type: str = "text") -> Tuple[IngestStats, float]:
        start_time = time.time()
        collection = self._open_collection(collection_name)
        stats = IngestStats()

        try:
            safe_source = source.replace('"', '""')
            collection.delete(filter=f'source = "{safe_source}"')
        except Exception:
            pass

        if not text.strip():
            return stats, 0.0

        if file_type.startswith('code_'):
            lang = file_type.replace('code_', '')
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix=f".{lang}", delete=False) as temp:
                temp.write(text)
                temp_path = Path(temp.name)
            try:
                code_chunks = chunk_code_ast(temp_path, lang, self.config.chunk_tokens)
                text_chunks =[(c, f'ast_{t}') for c, t in code_chunks]
            finally:
                temp_path.unlink()
        elif file_type == 'markdown':
            raw_chunks = chunk_text_semantic(text, self.config.chunk_tokens, self.config.chunk_overlap)
            text_chunks =[(c, 'semantic') for c in raw_chunks]
        else:
            raw_chunks = chunk_text_simple(text, self.config.chunk_tokens, self.config.chunk_overlap)
            text_chunks =[(c, 'simple') for c in raw_chunks]

        docs =[]
        seen_content = set()
        for idx, (chunk_text, _) in enumerate(text_chunks):
            if not chunk_text.strip(): continue
            if chunk_text in seen_content: continue
            seen_content.add(chunk_text)
                        
            try:
                doc = Doc(
                    id=self._generate_doc_id(source, idx),
                    vectors={
                        "text_embedding": self.embeddings.embed_dense(chunk_text),
                        "image_embedding": [0.0] * 512,
                        "sparse_embedding": self.embeddings.embed_sparse_doc(chunk_text),
                    },
                    fields={"content": chunk_text, "source": source, "chunk_id": idx, "file_type": file_type},
                )
                docs.append(doc)
            except Exception as e:
                stats.errors.append(f"Embed failed (chunk {idx}): {e}")

        if docs:
            result = collection.upsert(docs)
            stats.added = len(docs)
            if result and hasattr(result, 'code') and result.code != 0:
                stats.errors.append(f"Upsert failed: {result}")

        return stats, time.time() - start_time

    def _resolve_source(self, collection_name: str, filepath: Path) -> str:
        """Resolve the zrag:// source URI for a file."""
        try:
            info = self.list_collections_by_name(collection_name)
            if info and info.source_path:
                source_dir = Path(info.source_path).resolve()
                file_res = filepath.resolve()
                if file_res.is_relative_to(source_dir):
                    rel_path = file_res.relative_to(source_dir)
                    return f"zrag://{collection_name}/{str(rel_path).replace(os.sep, '/')}"
        except Exception:
            pass
        return f"zrag://{collection_name}/{filepath.name}"

    def _load_file_hashes(self, collection_name: str) -> Dict[str, str]:
        """Load file_hashes from metadata.json."""
        meta_file = self._get_collection_path(collection_name) / "metadata.json"
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                return json.load(f).get("file_hashes", {})
        return {}

    def _save_file_hash(self, collection_name: str, source: str, content_hash: str) -> None:
        """Save a single file hash to metadata.json."""
        meta_file = self._get_collection_path(collection_name) / "metadata.json"
        meta = {}
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                meta = json.load(f)
        hashes = meta.get("file_hashes", {})
        hashes[source] = content_hash
        meta["file_hashes"] = hashes
        with open(meta_file, 'w') as f:
            json.dump(meta, f, indent=2)

    def ingest_file(self, collection_name: str, filepath: Path, source: Optional[str] = None, force: bool = False) -> Tuple[IngestStats, float]:
        start_time = time.time()
        if not filepath.exists():
            return IngestStats(errors=[f"File not found: {filepath}"]), 0.0

        if not source:
            source = self._resolve_source(collection_name, filepath)

        file_type = get_file_type(filepath)

        if file_type == 'image':
            content_hash = hashlib.md5(filepath.read_bytes()).hexdigest()
            if not force:
                stored_hash = self._load_file_hashes(collection_name).get(source)
                if stored_hash == content_hash:
                    return IngestStats(unchanged=1), time.time() - start_time
                is_new = stored_hash is None
            else:
                is_new = True

            collection = self._open_collection(collection_name)
            try:
                safe_source = source.replace('"', '""')
                collection.delete(filter=f'source = "{safe_source}"')
            except Exception:
                pass
            try:
                doc = Doc(
                    id=self._generate_doc_id(source, 0),
                    vectors={
                        "text_embedding": [0.0] * self.config.text_dim,
                        "image_embedding": self.embeddings.get_image_embedding(filepath),
                        "sparse_embedding": {},
                    },
                    fields={"content": f"[IMAGE] {filepath.name}", "source": source, "chunk_id": 0, "file_type": file_type},
                )
                collection.upsert(doc)
            except Exception as e:
                return IngestStats(errors=[f"Image process failed: {e}"]), time.time() - start_time

            self._save_file_hash(collection_name, source, content_hash)
            return IngestStats(added=1 if is_new else 0, updated=0 if is_new else 1), time.time() - start_time

        # Text files: extract once, hash, then ingest if changed
        content = extract_text_from_file(filepath)
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()

        if not force:
            stored_hash = self._load_file_hashes(collection_name).get(source)
            if stored_hash == content_hash:
                return IngestStats(unchanged=1), time.time() - start_time
            is_new = stored_hash is None
        else:
            is_new = True

        stats, _ = self.ingest_text(collection_name, content, source, file_type)

        self._save_file_hash(collection_name, source, content_hash)

        result_stats = IngestStats(
            added=stats.added if is_new else 0,
            updated=stats.added if not is_new else 0,
            errors=stats.errors,
        )
        return result_stats, time.time() - start_time

    def ingest_url(self, collection_name: str, url: str, timeout: int = 10) -> Tuple[IngestStats, float]:
        start_time = time.time()
        content = extract_text_from_url(url, timeout)
        if not content.strip():
            return IngestStats(errors=[f"Failed to fetch content from {url}"]), time.time() - start_time
        return self.ingest_text(collection_name, content, url, file_type="markdown")

    # =========================================================================
    # Search & Retrieval Pipelines
    # =========================================================================

    def search_bm25(self, collection_name: str, query: str, top_k: Optional[int] = None, filter_expr: Optional[str] = None) -> Tuple[List[Doc], float]:
        start_time = time.time()
        collection = self._open_collection(collection_name)
        top_k = top_k or self.config.top_k

        sparse_vector = self.embeddings.embed_sparse_query(query)
        if not sparse_vector:
            return[], time.time() - start_time
            
        results = collection.query(
            vectors=zvec.VectorQuery(
                field_name="sparse_embedding",
                vector=sparse_vector,
            ),
            topk=top_k,
            filter=filter_expr,
            output_fields=["content", "source", "file_type", "chunk_id"],
        )
        
        self._inject_context(results)
        return results, time.time() - start_time
    
    def search_vector(self, collection_name: str, query: str, top_k: Optional[int] = None, filter_expr: Optional[str] = None, use_expansion: bool = True, use_hyde: bool = True, explain: bool = False) -> Tuple[List[Doc], float, Optional[Dict]]:
        start_time = time.time()
        collection = self._open_collection(collection_name)
        top_k = top_k or self.config.top_k

        if use_expansion:
            try:
                expansions = self._expand_query(query)
                vec_queries = expansions["vec"] or [query]
                if use_hyde and expansions["hyde"]: 
                    vec_queries.append(expansions["hyde"])
            except Exception:
                expansions = {"vec": [query], "hyde": ""}
                vec_queries = [query]
        else:
            expansions = {"vec": [query], "hyde": ""}
            vec_queries = [query]

        all_query_results = {}
        
        # Execute sequentially: Allows zvec to use its internal C++ query_threads without OS thrashing
        for idx, vec_query in enumerate(vec_queries):
            dense_vector = self.embeddings.embed_dense(vec_query)
            if dense_vector:
                res = collection.query(
                    vectors=[zvec.VectorQuery(
                        field_name="text_embedding",
                        vector=dense_vector,
                        param=zvec.HnswQueryParam(ef=self.config.hnsw_ef)
                    )],
                    topk=self.config.rerank_candidates,
                    filter=filter_expr,
                    output_fields=["content", "source", "file_type", "chunk_id"]
                )
                all_query_results[f"text_embedding_{idx}"] = res
        
        if not all_query_results:
            return[], time.time() - start_time, None

        # Delegate fusion to the zvec RerankFunction instance
        reranker = self.embeddings.get_reranker(query, top_n=top_k)
        final_docs = reranker.rerank(all_query_results)
        
        explanation = None
        if explain:
            explanation = {
                "vec_queries": vec_queries,
                "hyde": expansions.get("hyde", ""),
                "documents":[{"id": d.id, "score": getattr(d, 'score', 0.0), "rrf": d.fields.get('_rrf_score', 0.0), "ml": d.fields.get('_rerank_score', 0.0)} for d in final_docs]
            }

        self._inject_context(final_docs)
        return final_docs, time.time() - start_time, explanation

    def search_hybrid(self, collection_name: str, query: str, top_k: Optional[int] = None, filter_expr: Optional[str] = None, use_expansion: bool = True, use_hyde: bool = True, explain: bool = False) -> Tuple[List[Doc], float, Optional[Dict]]:
        start_time = time.time()
        collection = self._open_collection(collection_name)
        top_k = top_k or self.config.top_k
        
        if use_expansion:
            try:
                expansions = self._expand_query(query)
            except Exception:
                expansions = {"vec": [query], "lex": [query], "hyde": ""}
        else:
            expansions = {"vec": [query], "lex": [query], "hyde": ""}
        
        vec_queries = expansions["vec"] or [query]
        lex_queries = expansions["lex"] or [query]
        if use_hyde and expansions.get("hyde"):
            vec_queries.append(expansions["hyde"])

        all_query_results = {}
        
        # Execute BM25 sequentially
        for idx, lex_query in enumerate(lex_queries):
            sparse_vector = self.embeddings.embed_sparse_query(lex_query)
            if sparse_vector:
                res = collection.query(
                    vectors=[zvec.VectorQuery(
                        field_name="sparse_embedding", 
                        vector=sparse_vector, 
                    )],
                    topk=self.config.rerank_candidates,
                    filter=filter_expr,
                    output_fields=["content", "source", "file_type", "chunk_id"]
                )
                all_query_results[f"sparse_embedding_{idx}"] = res

        # Execute Vectors sequentially
        for idx, vec_query in enumerate(vec_queries):
            dense_vector = self.embeddings.embed_dense(vec_query)
            if dense_vector:
                res = collection.query(
                    vectors=[zvec.VectorQuery(
                        field_name="text_embedding", 
                        vector=dense_vector, 
                        param=zvec.HnswQueryParam(ef=self.config.hnsw_ef)
                    )],
                    topk=self.config.rerank_candidates,
                    filter=filter_expr,
                    output_fields=["content", "source", "file_type", "chunk_id"]
                )
                all_query_results[f"text_embedding_{idx}"] = res

        if not all_query_results:
            return[], time.time() - start_time, None

        # Pass independent pools directly to the position-aware blending reranker
        reranker = self.embeddings.get_reranker(query, top_n=top_k)
        final_docs = reranker.rerank(all_query_results)
         
        explanation = None
        if explain:
            explanation = {
                "lex_queries": lex_queries, 
                "vec_queries": vec_queries, 
                "hyde": expansions.get("hyde", ""), 
                "documents":[{"id": d.id, "score": getattr(d, 'score', 0.0), "rrf": d.fields.get('_rrf_score', 0.0), "ml": d.fields.get('_rerank_score', 0.0)} for d in final_docs]
            }
        
        self._inject_context(final_docs)
        return final_docs, time.time() - start_time, explanation

    def _expand_query(self, query: str) -> Dict[str, Any]:
        """Expand query using configured LLM API endpoint."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key="none", base_url=self.config.query_expansion_api_url)
            response = client.chat.completions.create(
                model=self.config.query_expansion_model,
                messages=[{"role": "user", "content": query}],
                max_tokens=512,
                temperature=0.0
            )
            lines = response.choices[0].message.content.split('\n')
            return {
                "lex": [line[4:].strip() for line in lines if line.startswith('lex:')],
                "vec": [line[4:].strip() for line in lines if line.startswith('vec:')],
                "hyde": next((line[5:].strip() for line in lines if line.startswith('hyde:')), "")
            }
        except Exception as e:
            print(f"⚠ Query expansion failed: {e}")
            return {"lex": [], "vec":[], "hyde": ""}
                    
    # =========================================================================
    # Context Management API Wrappers
    # =========================================================================

    def add_context(self, path: str, description: str) -> Dict[str, Any]:
        node = self.context_manager.add_context(path, description)
        return {'path': node.path, 'description': node.description, 'created_at': node.created_at, 'updated_at': node.updated_at}
    
    def remove_context(self, path: str) -> int:
        return self.context_manager.remove_context(path)
    
    def list_contexts(self, collection_name: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.context_manager.list_contexts(collection_name)
    
    def check_missing_context(self, collection_name: str) -> List[str]:
        collection = self._open_collection(collection_name)
        source_paths: Set[str] = set()
        results = collection.query(topk=1024, output_fields=["source"])
        if results:
            for doc in results:
                src = doc.fields.get("source")
                if src: source_paths.add(src)
        return self.context_manager.check_missing_context(list(source_paths))

    def get_by_id(self, collection_name: str, doc_id: str) -> Optional[Doc]:
        collection = self._open_collection(collection_name)
        result = collection.fetch(ids=doc_id)
        doc = result.get(doc_id)
        if doc: self._inject_context([doc])
        return doc
    
    def get_by_ids(self, collection_name: str, doc_ids: List[str]) -> Dict[str, Doc]:
        collection = self._open_collection(collection_name)
        docs = collection.fetch(ids=doc_ids)
        self._inject_context(list(docs.values()))
        return docs
    
    def get_by_glob(self, collection_name: str, pattern: str, top_k: int = 100) -> List[Doc]:
        collection = self._open_collection(collection_name)
        like_pattern = pattern.replace("*", "%").replace("?", "_")
        results = collection.query(filter=f'source like "{like_pattern}"', topk=top_k, output_fields=["content", "source", "chunk_id", "file_type"])
        self._inject_context(results)
        return results

    def _resolve_physical_path(self, collection_name: str, source_uri: str) -> Optional[Path]:
        if not source_uri.startswith("zrag://"): return Path(source_uri)
        parts = source_uri[7:].split("/", 1)
        if len(parts) < 2: return None
        col_name, rel_path = parts
        if col_name != collection_name: return None
        try:
            info = self.list_collections_by_name(collection_name)
            if info and info.source_path: return Path(info.source_path) / rel_path
        except Exception: pass
        return None

    def get_by_line_range(self, collection_name: str, source_path: str, start_line: int, end_line: int) -> Optional[Dict[str, Any]]:
        path = self._resolve_physical_path(collection_name, source_path)
        if path is None or not path.exists(): return None
        try:
            lines = path.read_text(encoding='utf-8').splitlines()
            start_idx = max(0, start_line - 1)
            end_idx = min(len(lines), end_line)
            if start_idx < end_idx:
                extracted = '\n'.join(lines[start_idx:end_idx])
                result = {
                    "id": f"{source_path}:{start_line}-{end_line}", "content": extracted, "source": source_path, "file_type": get_file_type(path),
                    "line_range": f"{start_line}-{end_line}", "start_line": start_line, "end_line": end_line,
                }
                contexts = self.context_manager.resolve_context(source_path)
                if contexts: result["context"] = " | ".join(contexts)
                return result
        except Exception as e:
            print(f"Error reading physical file {source_path}: {e}")
        return None

    def list_files_in_collection(self, collection_name: str, filter_pattern: Optional[str] = None) -> List[Dict[str, Any]]:
        collection = self._open_collection(collection_name)
        all_results = collection.query(topk=1024, output_fields=["source", "file_type", "chunk_id"])
        files = {}
        if all_results:
            for doc in all_results:
                source = doc.fields.get("source", "")
                if source not in files: files[source] = {"source": source, "file_type": doc.fields.get("file_type", "unknown"), "chunk_count": 0}
                files[source]["chunk_count"] += 1
        file_list = list(files.values())
        if filter_pattern:
            import fnmatch
            file_list =[f for f in file_list if fnmatch.fnmatch(f["source"], filter_pattern)]
        file_list.sort(key=lambda x: x["source"])
        return file_list

    def preload_resources(self) -> None:
        self.embeddings.preload_models()
    
    def get_status(self) -> Dict[str, Any]:
        return {"daemon_running": True, "collections_loaded": len(self._collections), "models_loaded": (self.embeddings._dense_embed is not None and self.embeddings._sparse_embed_query is not None and self.embeddings._sparse_embed_doc is not None)}
