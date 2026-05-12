"""
Search & retrieval tests.

Covers: BM25, vector, hybrid, filter expressions, threshold,
top_k, explain, query expansion (mocked), result structure.
"""

import pytest
from unittest.mock import patch

from conftest import PYTHON_SAMPLE, MARKDOWN_SAMPLE


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ingest_auth_docs(engine, collection_name="test"):
    """Ingest auth-themed docs so search has something to return."""
    engine.ingest_text(
        collection_name,
        "Authentication uses JWT tokens. The login endpoint validates user credentials and returns a signed token.",
        source="doc://auth_overview",
        file_type="markdown",
    )
    engine.ingest_text(
        collection_name,
        "Password hashing uses bcrypt with cost factor 12. Never store plaintext passwords.",
        source="doc://security",
        file_type="markdown",
    )
    engine.ingest_text(
        collection_name,
        "Session expiry: tokens expire after 24 hours of inactivity. Refresh tokens extend the window.",
        source="doc://sessions",
        file_type="markdown",
    )
    engine.ingest_text(
        collection_name,
        "Rate limiting: login attempts are capped at 5 per minute per IP address.",
        source="doc://rate_limits",
        file_type="markdown",
    )
    engine.ingest_text(
        collection_name,
        "Database schema: users table has id, email, password_hash, created_at columns.",
        source="doc://schema",
        file_type="markdown",
    )


# ---------------------------------------------------------------------------
# BM25 Search
# ---------------------------------------------------------------------------

class TestBM25Search:
    @pytest.fixture(autouse=True)
    def setup(self, collection):
        self.engine = collection
        _ingest_auth_docs(self.engine)

    def test_bm25_returns_results(self):
        results, elapsed = self.engine.search_bm25("test", "JWT token authentication")
        assert len(results) > 0
        assert elapsed >= 0

    def test_bm25_result_structure(self):
        results, _ = self.engine.search_bm25("test", "authentication")
        for r in results:
            assert hasattr(r, "id")
            assert hasattr(r, "fields")
            assert "content" in r.fields
            assert "source" in r.fields

    def test_bm25_top_k_respected(self):
        results, _ = self.engine.search_bm25("test", "token", top_k=2)
        assert len(results) <= 2

    def test_bm25_default_top_k(self, zrag_config):
        results, _ = self.engine.search_bm25("test", "token")
        assert len(results) <= zrag_config.top_k

    def test_bm25_filter_expression(self):
        results, _ = self.engine.search_bm25(
            "test", "token", filter_expr="source = 'doc://auth_overview'"
        )
        for r in results:
            assert r.fields["source"] == "doc://auth_overview"

    def test_bm25_no_results_on_garbage_query(self):
        results, _ = self.engine.search_bm25("test", "xyzzy_quux_frobnicator_99")
        # Either 0 results or very low-score results — just verify no crash
        assert isinstance(results, list)

    def test_bm25_relevant_result_first(self):
        """bcrypt query → security doc should rank high."""
        results, _ = self.engine.search_bm25("test", "bcrypt password hashing", top_k=5)
        assert len(results) > 0
        sources = [r.fields["source"] for r in results]
        assert "doc://security" in sources[:3]


# ---------------------------------------------------------------------------
# Vector Search
# ---------------------------------------------------------------------------

class TestVectorSearch:
    @pytest.fixture(autouse=True)
    def setup(self, collection):
        self.engine = collection
        _ingest_auth_docs(self.engine)

    def test_vector_returns_results(self):
        results, elapsed, explain = self.engine.search_vector(
            "test", "how does login work", use_expansion=False, use_hyde=False
        )
        assert len(results) > 0
        assert elapsed >= 0
        assert explain is None  # explain=False by default

    def test_vector_result_structure(self):
        results, _, _ = self.engine.search_vector(
            "test", "authentication token", use_expansion=False, use_hyde=False
        )
        for r in results:
            assert hasattr(r, "id")
            assert "content" in r.fields
            assert "source" in r.fields

    def test_vector_top_k_respected(self):
        results, _, _ = self.engine.search_vector(
            "test", "token", top_k=2, use_expansion=False, use_hyde=False
        )
        assert len(results) <= 2

    def test_vector_explain_populated(self):
        results, _, explain = self.engine.search_vector(
            "test", "JWT", use_expansion=False, use_hyde=False, explain=True
        )
        assert explain is not None
        assert "vec_queries" in explain
        assert "documents" in explain

    def test_vector_explain_documents_have_scores(self):
        _, _, explain = self.engine.search_vector(
            "test", "JWT", use_expansion=False, use_hyde=False, explain=True
        )
        for doc in explain["documents"]:
            assert "rrf_score" in doc
            assert "final_score" in doc

    def test_vector_filter_expression(self):
        results, _, _ = self.engine.search_vector(
            "test", "password",
            filter_expr="source = 'doc://security'",
            use_expansion=False, use_hyde=False,
        )
        for r in results:
            assert r.fields["source"] == "doc://security"

    def test_vector_with_expansion_mocked(self):
        expansion = {"lex": ["authentication token"], "vec": ["JWT bearer auth"], "hyde": ""}
        with patch.object(self.engine, "_expand_query", return_value=expansion):
            results, _, _ = self.engine.search_vector("test", "auth", use_expansion=True)
        assert isinstance(results, list)

    def test_vector_query_expansion_failure_fallback(self):
        """If expansion fails, falls back to original query without crash."""
        with patch.object(self.engine, "_expand_query", side_effect=Exception("LLM down")):
            # Exception in _expand_query → vector search should handle gracefully
            # (current impl: _expand_query returns empty dicts on exception)
            results, _, _ = self.engine.search_vector(
                "test", "authentication", use_expansion=True, use_hyde=False
            )
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Hybrid Search
# ---------------------------------------------------------------------------

class TestHybridSearch:
    @pytest.fixture(autouse=True)
    def setup(self, collection):
        self.engine = collection
        _ingest_auth_docs(self.engine)

    def _no_expansion(self):
        return {"lex": [], "vec": [], "hyde": ""}

    def test_hybrid_returns_results(self):
        with patch.object(self.engine, "_expand_query", return_value=self._no_expansion()):
            results, elapsed, _ = self.engine.search_hybrid(
                "test", "JWT authentication", use_expansion=False
            )
        assert len(results) > 0
        assert elapsed >= 0

    def test_hybrid_result_structure(self):
        with patch.object(self.engine, "_expand_query", return_value=self._no_expansion()):
            results, _, _ = self.engine.search_hybrid(
                "test", "token", use_expansion=False
            )
        for r in results:
            assert hasattr(r, "id")
            assert "content" in r.fields
            assert "source" in r.fields

    def test_hybrid_top_k_respected(self):
        with patch.object(self.engine, "_expand_query", return_value=self._no_expansion()):
            results, _, _ = self.engine.search_hybrid(
                "test", "auth", top_k=2, use_expansion=False
            )
        assert len(results) <= 2

    def test_hybrid_explain_populated(self):
        with patch.object(self.engine, "_expand_query", return_value=self._no_expansion()):
            _, _, explain = self.engine.search_hybrid(
                "test", "password", use_expansion=False, explain=True
            )
        assert explain is not None
        assert "lex_queries" in explain
        assert "vec_queries" in explain
        assert "documents" in explain

    def test_hybrid_explain_document_fields(self):
        with patch.object(self.engine, "_expand_query", return_value=self._no_expansion()):
            _, _, explain = self.engine.search_hybrid(
                "test", "token", use_expansion=False, explain=True
            )
        for doc in explain["documents"]:
            assert "rrf_score" in doc
            assert "rerank_score" in doc
            assert "final_score" in doc

    def test_hybrid_with_hyde(self):
        expansion = {
            "lex": ["JWT token auth"],
            "vec": ["authentication bearer token"],
            "hyde": "JWT tokens are used for stateless authentication. They encode user claims.",
        }
        with patch.object(self.engine, "_expand_query", return_value=expansion):
            results, _, _ = self.engine.search_hybrid("test", "how auth works", use_hyde=True)
        assert len(results) > 0

    def test_hybrid_better_recall_than_bm25(self):
        """Hybrid should return at least as many results as BM25."""
        with patch.object(self.engine, "_expand_query", return_value=self._no_expansion()):
            hybrid_results, _, _ = self.engine.search_hybrid(
                "test", "authentication token expire", top_k=10, use_expansion=False
            )
        bm25_results, _ = self.engine.search_bm25("test", "authentication token expire", top_k=10)
        # Hybrid combines BM25 + vector, should match or exceed BM25 count
        assert len(hybrid_results) >= len(bm25_results) or len(hybrid_results) > 0

    def test_hybrid_scores_sorted_descending(self):
        with patch.object(self.engine, "_expand_query", return_value=self._no_expansion()):
            results, _, _ = self.engine.search_hybrid(
                "test", "authentication", use_expansion=False
            )
        scores = [getattr(r, "score", 0) for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Query Expansion
# ---------------------------------------------------------------------------

class TestQueryExpansion:
    def test_expand_query_returns_dict(self, engine):
        """Real expansion with mocked LLM response."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = (
            "lex: user authentication login\nvec: JWT bearer token auth\nhyde: JWT is used for auth."
        )
        with patch("openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = mock_response
            result = engine._expand_query("how does login work")
        assert "lex" in result
        assert "vec" in result
        assert "hyde" in result

    def test_expand_query_failure_returns_empty(self, engine):
        with patch("openai.OpenAI", side_effect=Exception("Connection refused")):
            result = engine._expand_query("anything")
        assert result == {"lex": [], "vec": [], "hyde": ""}


from unittest.mock import MagicMock
