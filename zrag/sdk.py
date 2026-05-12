"""
Python SDK for zrag.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

from zrag.core import ZragEngine
from zrag.formatters import OutputFormat, OutputFormatter


class SearchStrategy(str, Enum):
    BM25 = "bm25"
    VECTOR = "vector"
    HYBRID = "hybrid"


@dataclass
class SearchResult:
    id: str
    score: Optional[float]
    content: str
    source: str
    chunk_id: Optional[int]
    file_type: Optional[str]
    context: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class LineRangeResult:
    id: str
    content: str
    source: str
    line_range: str
    start_line: int
    end_line: int
    file_type: Optional[str]
    context: Optional[str] = None


class ZragSDK:
    """Well-typed SDK strictly wrapping the ZragEngine."""
    
    def __init__(self, engine: ZragEngine):
        self.engine = engine
        self.formatter = OutputFormatter()
    
    def search(
        self,
        collection_name: str,
        query: str,
        strategy: SearchStrategy = SearchStrategy.HYBRID,
        top_k: int = 10,
        filter_expr: Optional[str] = None,
        use_expansion: bool = True,
        use_hyde: bool = True,
    ) -> List[SearchResult]:
        if strategy == SearchStrategy.BM25:
            results, _ = self.engine.search_bm25(collection_name, query, top_k, filter_expr)
        elif strategy == SearchStrategy.VECTOR:
            results, _, _ = self.engine.search_vector(collection_name, query, top_k, filter_expr, use_expansion, use_hyde)
        elif strategy == SearchStrategy.HYBRID:
            results, _, _ = self.engine.search_hybrid(collection_name, query, top_k, filter_expr, use_expansion, use_hyde)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
        return[self._doc_to_search_result(doc) for doc in results]
    
    def hybrid_query(
        self,
        collection_name: str,
        query: str,
        top_k: int = 10,
        filter_expr: Optional[str] = None,
        use_expansion: bool = True,
        use_hyde: bool = True,
    ) -> List[SearchResult]:
        return self.search(collection_name, query, SearchStrategy.HYBRID, top_k, filter_expr, use_expansion, use_hyde)
    
    def get(self, collection_name: str, identifier: str, **kwargs) -> Union[SearchResult, LineRangeResult, List[SearchResult]]:
        path_part = identifier
        range_part = None
        if ":" in identifier:
            parts = identifier.rsplit(":", 1)
            if len(parts) == 2 and "-" in parts[1]:
                path_part, range_part = parts
                
        # Check if path_part is a doc ID
        doc = self.engine.get_by_id(collection_name, path_part)
        source_path = doc.fields.get("source", path_part) if doc else path_part

        if range_part and "-" in range_part:
            start, end = range_part.split("-")
            try:
                result = self.engine.get_by_line_range(collection_name, source_path, int(start), int(end))
                if result:
                    return LineRangeResult(**{k:v for k,v in result.items() if k != 'chunk_id'})
            except ValueError:
                pass
        
        if doc:
            return self._doc_to_search_result(doc)
        
        if identifier.startswith("glob:"):
            results = self.engine.get_by_glob(collection_name, identifier[5:], kwargs.get("top_k", 100))
            return[self._doc_to_search_result(doc) for doc in results]
        
        if "," in identifier:
            doc_ids = [id.strip() for id in identifier.split(",")]
            docs = self.engine.get_by_ids(collection_name, doc_ids)
            return[self._doc_to_search_result(doc) for doc in docs.values()]
        
        # Fallback: exact match by source path
        safe_path = path_part.replace('"', '""')
        results = self.engine._open_collection(collection_name).query(
            filter=f'source = "{safe_path}"',
            topk=kwargs.get("top_k", 1024),
            output_fields=["content", "source", "chunk_id", "file_type"],
        )
        if results:
            self.engine._inject_context(results)
            return[self._doc_to_search_result(d) for d in results]
            
        return None
    
    def _doc_to_search_result(self, doc) -> SearchResult:
        fields = doc.fields.copy()
        return SearchResult(
            id=doc.id,
            score=getattr(doc, 'score', None),
            content=fields.pop("content", ""),
            source=fields.pop("source", ""),
            chunk_id=fields.pop("chunk_id", None),
            file_type=fields.pop("file_type", None),
            context=fields.pop("context", None),
            metadata=fields,
        )
