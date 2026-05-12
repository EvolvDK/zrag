"""
Ingestion pipeline tests.

Covers: text, file (md/py/go/rs/pdf/docx/image), URL, update/sync,
git pull integration, upsert idempotency, dedup, error handling.
"""

import textwrap
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from conftest import PYTHON_SAMPLE, MARKDOWN_SAMPLE, GO_SAMPLE, RUST_SAMPLE


# ---------------------------------------------------------------------------
# Text Ingestion
# ---------------------------------------------------------------------------

class TestIngestText:
    def test_ingest_basic_text(self, collection):
        stats, elapsed = collection.ingest_text("test", "Hello world content.", source="test://hello")
        assert stats.added >= 1
        assert elapsed >= 0

    def test_ingest_empty_text_no_docs(self, collection):
        stats, elapsed = collection.ingest_text("test", "   ", source="test://empty")
        assert stats.added == 0

    def test_ingest_markdown_type(self, collection):
        stats, _ = collection.ingest_text("test", MARKDOWN_SAMPLE, source="test://md", file_type="markdown")
        assert stats.added >= 1

    def test_ingest_python_type(self, collection):
        stats, _ = collection.ingest_text("test", PYTHON_SAMPLE, source="test://py", file_type="code_python")
        assert stats.added >= 1

    def test_ingest_returns_elapsed_float(self, collection):
        _, elapsed = collection.ingest_text("test", "some text", source="test://t")
        assert isinstance(elapsed, float)

    def test_ingest_purges_old_chunks(self, collection):
        """Re-ingesting same source replaces old chunks — no ghost duplicates."""
        source = "test://evolving"
        collection.ingest_text("test", "Version 1 content.", source=source)
        collection.ingest_text("test", "Version 2 completely different.", source=source)

        # Search for V1 term — should NOT appear (deleted)
        results, _ = collection.search_bm25("test", "Version 1 content")
        sources = [r.fields.get("source") for r in results]
        # We check: no result for source contains old content "Version 1"
        for r in results:
            if r.fields.get("source") == source:
                assert "Version 2" in r.fields.get("content", ""), "Old chunk not purged"

    def test_ingest_no_duplicate_chunks(self, collection):
        """Same text ingested → chunk IDs deduplicated in ingest_text."""
        text = "Unique text content for dedup test."
        collection.ingest_text("test", text * 5, source="test://dedup_text")
        # Verify through list_files: only 1 file entry
        files = collection.list_files_in_collection("test")
        dedup_files = [f for f in files if "dedup_text" in f["source"]]
        assert len(dedup_files) == 1

    def test_ingest_special_chars_in_source(self, collection):
        """Source with single quotes must not crash SQL filter."""
        stats, _ = collection.ingest_text(
            "test", "Content here.", source="path/to/O'Brien/file.md"
        )
        assert stats.errors == []


# ---------------------------------------------------------------------------
# File Ingestion
# ---------------------------------------------------------------------------

class TestIngestFile:
    def test_ingest_python_file(self, collection, tmp_path):
        f = tmp_path / "auth.py"
        f.write_text(PYTHON_SAMPLE)
        stats, _ = collection.ingest_file("test", f)
        assert stats.added >= 1
        assert stats.errors == []

    def test_ingest_markdown_file(self, collection, tmp_path):
        f = tmp_path / "guide.md"
        f.write_text(MARKDOWN_SAMPLE)
        stats, _ = collection.ingest_file("test", f)
        assert stats.added >= 1

    def test_ingest_go_file(self, collection, tmp_path):
        f = tmp_path / "session.go"
        f.write_text(GO_SAMPLE)
        stats, _ = collection.ingest_file("test", f)
        assert stats.added >= 1

    def test_ingest_rust_file(self, collection, tmp_path):
        f = tmp_path / "store.rs"
        f.write_text(RUST_SAMPLE)
        stats, _ = collection.ingest_file("test", f)
        assert stats.added >= 1

    def test_ingest_txt_file(self, collection, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_text("Plain text content for testing.\nSecond line.")
        stats, _ = collection.ingest_file("test", f)
        assert stats.added >= 1

    def test_ingest_missing_file_returns_error(self, collection, tmp_path):
        ghost = tmp_path / "ghost.md"
        stats, _ = collection.ingest_file("test", ghost)
        assert len(stats.errors) >= 1
        assert "not found" in stats.errors[0].lower()

    def test_ingest_uses_filepath_as_default_source(self, collection, tmp_path):
        f = tmp_path / "auto_source.py"
        f.write_text(PYTHON_SAMPLE)
        collection.ingest_file("test", f)
        files = collection.list_files_in_collection("test")
        sources = [fi["source"] for fi in files]
        assert "zrag://test/auto_source.py" in sources

    def test_ingest_custom_source_override(self, collection, tmp_path):
        f = tmp_path / "override.py"
        f.write_text(PYTHON_SAMPLE)
        collection.ingest_file("test", f, source="custom://source")
        files = collection.list_files_in_collection("test")
        sources = [fi["source"] for fi in files]
        assert "custom://source" in sources

    def test_ingest_upsert_idempotent(self, collection, tmp_path):
        """Ingesting same file twice should not double chunk count."""
        f = tmp_path / "idem.py"
        f.write_text(PYTHON_SAMPLE)
        source = "zrag://test/idem.py"
        collection.ingest_file("test", f)
        files_before = collection.list_files_in_collection("test")
        count_before = next(fi["chunk_count"] for fi in files_before if fi["source"] == source)

        collection.ingest_file("test", f)  # second ingest
        files_after = collection.list_files_in_collection("test")
        count_after = next(fi["chunk_count"] for fi in files_after if fi["source"] == source)
        assert count_after == count_before, "Upsert created duplicate chunks"

    def test_ingest_image_file(self, collection, tmp_path):
        """Image files: should produce 1 doc with [IMAGE] prefix, no crash."""
        from PIL import Image
        img_path = tmp_path / "test_image.png"
        img = Image.new("RGB", (64, 64), color=(128, 0, 128))
        img.save(img_path)
        stats, _ = collection.ingest_file("test", img_path)
        assert stats.added == 1
        assert stats.errors == []


# ---------------------------------------------------------------------------
# URL Ingestion
# ---------------------------------------------------------------------------

class TestIngestUrl:
    def test_ingest_url_mock(self, collection):
        """Mock HTTP fetch → verify ingestion pipeline works end-to-end."""
        html = "<html><body><h1>Auth Guide</h1><p>JWT tokens expire after 24 hours.</p></body></html>"
        with patch("zrag.file_processing.fetch_web_content", return_value=html):
            stats, elapsed = collection.ingest_url("test", "https://example.com/auth")
        assert stats.added >= 1

    def test_ingest_url_empty_content_returns_error(self, collection):
        with patch("zrag.file_processing.fetch_web_content", return_value=""):
            stats, _ = collection.ingest_url("test", "https://example.com/empty")
        assert len(stats.errors) >= 1

    def test_ingest_url_source_is_url(self, collection):
        url = "https://example.com/docs"
        html = "<html><body><p>Some content about authentication.</p></body></html>"
        with patch("zrag.file_processing.fetch_web_content", return_value=html):
            collection.ingest_url("test", url)
        files = collection.list_files_in_collection("test")
        sources = [f["source"] for f in files]
        assert url in sources


# ---------------------------------------------------------------------------
# Update / Sync
# ---------------------------------------------------------------------------

class TestUpdateCollection:
    def test_update_no_source_path_error(self, engine):
        engine.create_collection("nosrc")
        stats, _ = engine.update_collection("nosrc")
        assert len(stats.errors) >= 1
        assert "source_path" in stats.errors[0].lower()

    def test_update_scans_source_files(self, engine, source_repo):
        engine.create_collection(
            "sync_col",
            source_path=str(source_repo),
            mask=None,
        )
        # Files already ingested by create_collection; verify
        files = engine.list_files_in_collection("sync_col")
        assert len(files) >= 4  # py, md, go, rs, txt

    def test_update_new_file_detected(self, engine, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "a.md").write_text(MARKDOWN_SAMPLE)
        engine.create_collection("incremental", source_path=str(repo))

        # Add new file, trigger update
        (repo / "b.md").write_text("# New document\n\nNew content here.")
        stats, _ = engine.update_collection("incremental")
        assert stats.added >= 1

    def test_update_with_pull_git_not_repo(self, engine, tmp_path):
        """Git pull on non-repo dir: should skip silently, not crash."""
        repo = tmp_path / "notgit"
        repo.mkdir()
        (repo / "a.md").write_text(MARKDOWN_SAMPLE)
        engine.create_collection("nogit", source_path=str(repo))
        stats, _ = engine.update_collection("nogit", pull=True)
        # No crash; pull errors are acceptable but ingestion continues
        assert isinstance(stats.added, int)

    def test_update_with_mask(self, engine, tmp_path):
        repo = tmp_path / "masked_repo"
        repo.mkdir()
        (repo / "code.py").write_text(PYTHON_SAMPLE)
        (repo / "doc.md").write_text(MARKDOWN_SAMPLE)

        engine.create_collection("masked", source_path=str(repo), mask="**/*.md")
        files = engine.list_files_in_collection("masked")
        assert all(f["source"].endswith(".md") for f in files), "Mask not applied"

    def test_update_updates_timestamp(self, engine, tmp_path):
        import time, json
        repo = tmp_path / "tsrepo"
        repo.mkdir()
        (repo / "a.md").write_text(MARKDOWN_SAMPLE)
        engine.create_collection("ts_update", source_path=str(repo))

        meta_file = engine._get_collection_path("ts_update") / "metadata.json"
        before = json.loads(meta_file.read_text())["updated_at"]
        time.sleep(0.1)
        engine.update_collection("ts_update")
        after = json.loads(meta_file.read_text())["updated_at"]
        assert after > before
