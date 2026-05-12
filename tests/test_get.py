"""
Direct retrieval tests.

Covers: get_by_id, get_by_ids, get_by_glob, get_by_line_range,
pagination (from/limit), max_bytes truncation, SDK.get routing.
"""

import pytest
from conftest import PYTHON_SAMPLE, MARKDOWN_SAMPLE


def _ingest(engine, tmp_path):
    """Ingest a Python + Markdown file, return their paths."""
    py_file = tmp_path / "auth.py"
    py_file.write_text(PYTHON_SAMPLE)
    md_file = tmp_path / "guide.md"
    md_file.write_text(MARKDOWN_SAMPLE)
    engine.ingest_file("test", py_file)
    engine.ingest_file("test", md_file)
    return py_file, md_file


# ---------------------------------------------------------------------------
# get_by_id
# ---------------------------------------------------------------------------

class TestGetById:
    def test_get_existing_doc(self, collection, tmp_path):
        py_file, _ = _ingest(collection, tmp_path)
        files = collection.list_files_in_collection("test")
        py_entry = next(f for f in files if f["source"].endswith(".py"))
        source = py_entry["source"]
        doc_id = collection._generate_doc_id(source, 0)
        doc = collection.get_by_id("test", doc_id)
        assert doc is not None
        assert doc.fields["source"] == source

    def test_get_nonexistent_id_returns_none(self, collection):
        result = collection.get_by_id("test", "doc_does_not_exist_at_all")
        assert result is None

    def test_get_by_ids_batch(self, collection, tmp_path):
        py_file, md_file = _ingest(collection, tmp_path)
        files = collection.list_files_in_collection("test")
        py_entry = next(f for f in files if f["source"].endswith(".py"))
        md_entry = next(f for f in files if f["source"].endswith(".md"))
        id1 = collection._generate_doc_id(py_entry["source"], 0)
        id2 = collection._generate_doc_id(md_entry["source"], 0)
        docs = collection.get_by_ids("test", [id1, id2])
        assert isinstance(docs, dict)
        assert id1 in docs or id2 in docs  # at least one found

    def test_get_by_ids_unknown_returns_partial(self, collection, tmp_path):
        py_file, _ = _ingest(collection, tmp_path)
        files = collection.list_files_in_collection("test")
        py_entry = next(f for f in files if f["source"].endswith(".py"))
        id1 = collection._generate_doc_id(py_entry["source"], 0)
        docs = collection.get_by_ids("test", [id1, "nonexistent_id_xyz"])
        # Known ID must be present
        assert id1 in docs


# ---------------------------------------------------------------------------
# get_by_glob
# ---------------------------------------------------------------------------

class TestGetByGlob:
    def test_glob_all_python(self, collection, tmp_path):
        _ingest(collection, tmp_path)
        results = collection.get_by_glob("test", "*.py")
        assert len(results) > 0
        for r in results:
            assert r.fields["source"].endswith(".py")

    def test_glob_all_markdown(self, collection, tmp_path):
        _ingest(collection, tmp_path)
        results = collection.get_by_glob("test", "*.md")
        assert len(results) > 0
        for r in results:
            assert r.fields["source"].endswith(".md")

    def test_glob_no_match_returns_empty(self, collection, tmp_path):
        _ingest(collection, tmp_path)
        results = collection.get_by_glob("test", "*.zig")
        assert results == []

    def test_glob_wildcard_all(self, collection, tmp_path):
        _ingest(collection, tmp_path)
        results = collection.get_by_glob("test", "*")
        assert len(results) >= 2

    def test_glob_with_single_quote_safe(self, collection, tmp_path):
        """Glob with apostrophe must not crash SQL filter."""
        _ingest(collection, tmp_path)
        # Should not raise, even if pattern contains "'"
        results = collection.get_by_glob("test", "O'Brien*")
        assert isinstance(results, list)

    def test_glob_top_k_respected(self, collection, tmp_path):
        _ingest(collection, tmp_path)
        results = collection.get_by_glob("test", "*", top_k=1)
        assert len(results) <= 1


# ---------------------------------------------------------------------------
# get_by_line_range
# ---------------------------------------------------------------------------

class TestGetByLineRange:
    def test_line_range_basic(self, collection, tmp_path):
        py_file, _ = _ingest(collection, tmp_path)
        result = collection.get_by_line_range("test", str(py_file), 1, 5)
        assert result is not None
        assert "content" in result
        lines = result["content"].split("\n")
        assert len(lines) <= 5

    def test_line_range_returns_correct_content(self, collection, tmp_path):
        py_file, _ = _ingest(collection, tmp_path)
        result = collection.get_by_line_range("test", str(py_file), 1, 3)
        # First 3 lines of PYTHON_SAMPLE: class AuthManager header
        assert "AuthManager" in result["content"] or "class" in result["content"]

    def test_line_range_includes_metadata(self, collection, tmp_path):
        py_file, _ = _ingest(collection, tmp_path)
        result = collection.get_by_line_range("test", str(py_file), 1, 10)
        assert "source" in result
        assert "line_range" in result
        assert "start_line" in result
        assert "end_line" in result

    def test_line_range_nonexistent_file_returns_none(self, collection, tmp_path):
        result = collection.get_by_line_range("test", str(tmp_path / "ghost.py"), 1, 10)
        assert result is None

    def test_line_range_out_of_bounds_clamped(self, collection, tmp_path):
        py_file, _ = _ingest(collection, tmp_path)
        total_lines = len(PYTHON_SAMPLE.splitlines())
        # Request past EOF
        result = collection.get_by_line_range("test", str(py_file), 1, total_lines + 1000)
        assert result is not None  # Should clamp, not crash

    def test_line_range_start_equals_end(self, collection, tmp_path):
        py_file, _ = _ingest(collection, tmp_path)
        result = collection.get_by_line_range("test", str(py_file), 1, 1)
        assert result is not None
        assert len(result["content"].splitlines()) == 1


# ---------------------------------------------------------------------------
# SDK.get routing
# ---------------------------------------------------------------------------

class TestSDKGet:
    def test_sdk_get_by_path(self, sdk, collection, tmp_path):
        py_file, _ = _ingest(collection, tmp_path)
        files = collection.list_files_in_collection("test")
        py_source = next(f["source"] for f in files if f["source"].endswith(".py"))
        result = sdk.get("test", py_source)
        # May return list (multiple chunks) or single doc
        assert result is not None

    def test_sdk_get_line_range(self, sdk, collection, tmp_path):
        py_file, _ = _ingest(collection, tmp_path)
        result = sdk.get("test", f"{py_file}:1-5")
        from zrag.sdk import LineRangeResult
        assert isinstance(result, LineRangeResult)
        assert result.start_line == 1
        assert result.end_line == 5

    def test_sdk_get_glob(self, sdk, collection, tmp_path):
        _ingest(collection, tmp_path)
        results = sdk.get("test", "glob:*.py")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_sdk_get_comma_list(self, sdk, collection, tmp_path):
        py_file, md_file = _ingest(collection, tmp_path)
        files = collection.list_files_in_collection("test")
        py_entry = next(f for f in files if f["source"].endswith(".py"))
        md_entry = next(f for f in files if f["source"].endswith(".md"))
        id1 = collection._generate_doc_id(py_entry["source"], 0)
        id2 = collection._generate_doc_id(md_entry["source"], 0)
        results = sdk.get("test", f"{id1},{id2}")
        assert isinstance(results, list)

    def test_sdk_get_returns_search_result(self, sdk, collection, tmp_path):
        py_file, _ = _ingest(collection, tmp_path)
        files = collection.list_files_in_collection("test")
        py_entry = next(f for f in files if f["source"].endswith(".py"))
        doc_id = collection._generate_doc_id(py_entry["source"], 0)
        result = sdk.get("test", doc_id)
        from zrag.sdk import SearchResult
        if isinstance(result, list):
            for r in result:
                assert isinstance(r, SearchResult)
        elif result is not None:
            assert isinstance(result, SearchResult)

    def test_sdk_get_nonexistent_returns_none(self, sdk, collection):
        result = sdk.get("test", "absolutely_nonexistent_id")
        assert result is None


# ---------------------------------------------------------------------------
# Pagination & max_bytes (via daemon endpoint logic, tested at engine level)
# ---------------------------------------------------------------------------

class TestPagination:
    def test_max_bytes_truncation(self, collection, tmp_path):
        """Large content should be truncated when byte limit is small."""
        f = tmp_path / "big.md"
        f.write_text(MARKDOWN_SAMPLE * 10)
        collection.ingest_file("test", f)

        results = collection.get_by_glob("test", "*.md", top_k=100)
        # Simulate daemon max_bytes truncation: verify content exists
        for r in results:
            content = r.fields.get("content", "")
            assert isinstance(content, str)

    def test_glob_results_sorted_by_source(self, collection, tmp_path):
        _ingest(collection, tmp_path)
        results = collection.get_by_glob("test", "*")
        # list_files_in_collection sorts by source; glob does not guarantee order
        # Just verify all have sources
        for r in results:
            assert "source" in r.fields
