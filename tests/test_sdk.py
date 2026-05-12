"""
SDK parity tests.

Verifies: ZragSDK wraps engine correctly, returns typed objects,
strategy enum routes correctly, all search strategies produce
SearchResult, SDK.get routing matches spec.
"""

import pytest
from unittest.mock import patch

from zrag.sdk import ZragSDK, SearchResult, LineRangeResult, SearchStrategy
from conftest import PYTHON_SAMPLE, MARKDOWN_SAMPLE


def _ingest_docs(engine, tmp_path):
    md = tmp_path / "guide.md"
    md.write_text(MARKDOWN_SAMPLE)
    py = tmp_path / "auth.py"
    py.write_text(PYTHON_SAMPLE)
    engine.ingest_file("test", md)
    engine.ingest_file("test", py)
    return md, py


# ---------------------------------------------------------------------------
# SearchResult type
# ---------------------------------------------------------------------------

class TestSearchResultType:
    def test_search_result_fields(self):
        r = SearchResult(
            id="x", score=0.9, content="text", source="file.md",
            chunk_id=0, file_type="markdown"
        )
        assert r.id == "x"
        assert r.score == 0.9
        assert r.content == "text"
        assert r.source == "file.md"
        assert r.chunk_id == 0
        assert r.file_type == "markdown"
        assert r.context is None
        assert r.metadata == {}

    def test_search_result_with_context(self):
        r = SearchResult(
            id="x", score=0.5, content="c", source="s",
            chunk_id=0, file_type="text", context="ctx"
        )
        assert r.context == "ctx"


# ---------------------------------------------------------------------------
# Strategy routing
# ---------------------------------------------------------------------------

class TestSearchStrategyRouting:
    @pytest.fixture(autouse=True)
    def setup(self, collection, tmp_path):
        self.sdk = ZragSDK(collection)
        self.engine = collection
        _ingest_docs(collection, tmp_path)

    def test_bm25_strategy(self):
        results = self.sdk.search("test", "JWT authentication", strategy=SearchStrategy.BM25)
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, SearchResult)

    def test_vector_strategy(self):
        results = self.sdk.search(
            "test", "token auth",
            strategy=SearchStrategy.VECTOR,
            use_expansion=False, use_hyde=False,
        )
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, SearchResult)

    def test_hybrid_strategy(self):
        with patch.object(self.engine, "_expand_query", return_value={"lex": [], "vec": [], "hyde": ""}):
            results = self.sdk.search(
                "test", "authentication",
                strategy=SearchStrategy.HYBRID,
                use_expansion=False,
            )
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, SearchResult)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            self.sdk.search("test", "auth", strategy="totally_wrong")

    def test_hybrid_query_shortcut(self):
        with patch.object(self.engine, "_expand_query", return_value={"lex": [], "vec": [], "hyde": ""}):
            results = self.sdk.hybrid_query("test", "auth", use_expansion=False)
        assert isinstance(results, list)

    def test_top_k_respected(self):
        results = self.sdk.search("test", "auth", strategy=SearchStrategy.BM25, top_k=1)
        assert len(results) <= 1

    def test_filter_expr_passed_through(self, tmp_path):
        md = tmp_path / "guide2.md"
        md.write_text(MARKDOWN_SAMPLE)
        self.engine.ingest_file("test", md)
        source = "zrag://test/guide2.md"
        safe_source = source.replace("'", "''")
        results = self.sdk.search(
            "test", "authentication",
            strategy=SearchStrategy.BM25,
            filter_expr=f"source = '{safe_source}'",
        )
        for r in results:
            assert r.source == source


# ---------------------------------------------------------------------------
# SDK.get routing
# ---------------------------------------------------------------------------

class TestSDKGetRouting:
    @pytest.fixture(autouse=True)
    def setup(self, collection, tmp_path):
        self.sdk = ZragSDK(collection)
        self.engine = collection
        self.md, self.py = _ingest_docs(collection, tmp_path)
        # Sources are always zrag:// URIs
        self.md_source = "zrag://test/guide.md"
        self.py_source = "zrag://test/auth.py"

    def test_get_by_doc_id(self):
        doc_id = self.engine._generate_doc_id(self.md_source, 0)
        result = self.sdk.get("test", doc_id)
        if result is not None:
            assert isinstance(result, SearchResult)

    def test_get_by_path_returns_chunks(self):
        result = self.sdk.get("test", self.md_source)
        assert result is not None
        if isinstance(result, list):
            assert len(result) >= 1

    def test_get_line_range(self):
        result = self.sdk.get("test", f"{self.py}:1-10")
        assert isinstance(result, LineRangeResult)
        assert result.start_line == 1
        assert result.end_line == 10

    def test_get_glob(self):
        results = self.sdk.get("test", "glob:*.md")
        assert isinstance(results, list)
        assert len(results) >= 1
        for r in results:
            assert isinstance(r, SearchResult)

    def test_get_comma_ids(self):
        id1 = self.engine._generate_doc_id(self.md_source, 0)
        id2 = self.engine._generate_doc_id(self.py_source, 0)
        results = self.sdk.get("test", f"{id1},{id2}")
        assert isinstance(results, list)

    def test_get_nonexistent_returns_none(self):
        result = self.sdk.get("test", "totally_fake_id_not_in_db")
        assert result is None

    def test_search_result_content_not_empty(self, tmp_path):
        result = self.sdk.get("test", self.md_source)
        if isinstance(result, list):
            for r in result:
                assert isinstance(r.content, str)
                assert len(r.content) > 0
        elif isinstance(result, SearchResult):
            assert len(result.content) > 0


# ---------------------------------------------------------------------------
# SDK/Engine parity: same results
# ---------------------------------------------------------------------------

class TestSDKEngineParity:
    """SDK must return identical data to underlying engine calls."""

    @pytest.fixture(autouse=True)
    def setup(self, collection, tmp_path):
        self.sdk = ZragSDK(collection)
        self.engine = collection
        _ingest_docs(collection, tmp_path)

    def test_bm25_same_count(self):
        sdk_results = self.sdk.search("test", "JWT", strategy=SearchStrategy.BM25, top_k=5)
        eng_results, _ = self.engine.search_bm25("test", "JWT", top_k=5)
        assert len(sdk_results) == len(eng_results)

    def test_bm25_same_sources(self):
        sdk_results = self.sdk.search("test", "authentication", strategy=SearchStrategy.BM25, top_k=5)
        eng_results, _ = self.engine.search_bm25("test", "authentication", top_k=5)
        sdk_sources = {r.source for r in sdk_results}
        eng_sources = {r.fields["source"] for r in eng_results}
        assert sdk_sources == eng_sources

    def test_vector_same_count(self):
        sdk_results = self.sdk.search(
            "test", "token auth", strategy=SearchStrategy.VECTOR,
            top_k=5, use_expansion=False, use_hyde=False,
        )
        eng_results, _, _ = self.engine.search_vector(
            "test", "token auth", top_k=5, use_expansion=False, use_hyde=False
        )
        assert len(sdk_results) == len(eng_results)
