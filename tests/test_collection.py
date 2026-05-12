"""
Collection management tests.

Covers: create, list, remove, rename, inspect, optimize, ls.
"""

import pytest
from pathlib import Path

from conftest import PYTHON_SAMPLE, MARKDOWN_SAMPLE


class TestCollectionCreate:
    def test_create_basic(self, engine):
        info = engine.create_collection("mycol")
        assert info.name == "mycol"
        assert info.document_count == 0

    def test_create_with_description(self, engine):
        info = engine.create_collection("described", description="Test description")
        assert info.description == "Test description"

    def test_create_duplicate_raises(self, engine):
        engine.create_collection("dup")
        with pytest.raises(ValueError, match="already exists"):
            engine.create_collection("dup")

    def test_create_persists_on_disk(self, engine, zrag_config):
        engine.create_collection("persistent")
        col_path = zrag_config.collections_dir / "persistent"
        assert col_path.exists()

    def test_create_writes_metadata(self, engine, zrag_config):
        import json
        engine.create_collection("meta_col", description="desc", mask="**/*.py")
        meta = json.loads((zrag_config.collections_dir / "meta_col" / "metadata.json").read_text())
        assert meta["mask"] == "**/*.py"
        assert meta["created_at"] > 0
        # Description is stored in context manager, not metadata.json
        ctx = engine.context_manager.get_context("zrag://meta_col")
        assert ctx is not None
        assert ctx.description == "desc"

    def test_create_with_source_path_auto_ingests(self, engine, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "sample.md").write_text(MARKDOWN_SAMPLE)
        info = engine.create_collection("autoingest", source_path=str(repo))
        # After auto-ingest, document_count > 0
        assert info.document_count > 0


class TestCollectionList:
    def test_list_empty(self, engine):
        cols = engine.list_collections()
        assert isinstance(cols, list)

    def test_list_after_create(self, engine):
        engine.create_collection("col_a")
        engine.create_collection("col_b")
        names = [c.name for c in engine.list_collections()]
        assert "col_a" in names
        assert "col_b" in names

    def test_list_by_name_found(self, engine):
        engine.create_collection("findme")
        info = engine.list_collections_by_name("findme")
        assert info.name == "findme"

    def test_list_by_name_not_found_raises(self, engine):
        with pytest.raises(ValueError):
            engine.list_collections_by_name("does_not_exist")

    def test_list_returns_size_bytes(self, engine):
        engine.create_collection("sized")
        info = engine.list_collections_by_name("sized")
        assert info.size_bytes >= 0


class TestCollectionRemove:
    def test_remove_existing(self, engine, zrag_config):
        engine.create_collection("toremove")
        engine.remove_collection("toremove")
        col_path = zrag_config.collections_dir / "toremove"
        assert not col_path.exists()

    def test_remove_nonexistent_raises(self, engine):
        with pytest.raises(ValueError, match="does not exist"):
            engine.remove_collection("ghost")

    def test_remove_clears_from_cache(self, engine):
        engine.create_collection("cached")
        engine._open_collection("cached")  # put in cache
        engine.remove_collection("cached")
        assert "cached" not in engine._collections

    def test_removed_not_in_list(self, engine):
        engine.create_collection("gone")
        engine.remove_collection("gone")
        names = [c.name for c in engine.list_collections()]
        assert "gone" not in names


class TestCollectionRename:
    def test_rename_basic(self, engine, zrag_config):
        engine.create_collection("old_name")
        info = engine.rename_collection("old_name", "new_name")
        assert info.name == "new_name"
        assert (zrag_config.collections_dir / "new_name").exists()
        assert not (zrag_config.collections_dir / "old_name").exists()

    def test_rename_updates_timestamp(self, engine):
        import time
        engine.create_collection("ts_col")
        before = engine.list_collections_by_name("ts_col").updated_at
        time.sleep(0.05)
        engine.rename_collection("ts_col", "ts_col_renamed")
        after = engine.list_collections_by_name("ts_col_renamed").updated_at
        assert after >= before

    def test_rename_nonexistent_raises(self, engine):
        with pytest.raises(ValueError):
            engine.rename_collection("no_such", "whatever")

    def test_rename_to_existing_raises(self, engine):
        engine.create_collection("r_a")
        engine.create_collection("r_b")
        with pytest.raises(ValueError):
            engine.rename_collection("r_a", "r_b")

    def test_rename_clears_old_from_cache(self, engine):
        engine.create_collection("old_cached")
        engine._open_collection("old_cached")
        engine.rename_collection("old_cached", "new_cached")
        assert "old_cached" not in engine._collections


class TestCollectionInspect:
    def test_inspect_returns_schema(self, engine):
        engine.create_collection("inspect_me")
        schema = engine.inspect_collection("inspect_me")
        assert schema["name"] == "inspect_me"
        assert "fields" in schema
        assert "vectors" in schema

    def test_inspect_has_required_fields(self, engine):
        engine.create_collection("field_check")
        schema = engine.inspect_collection("field_check")
        field_names = [f["name"] for f in schema["fields"]]
        assert "content" in field_names
        assert "source" in field_names
        assert "chunk_id" in field_names
        assert "file_type" in field_names

    def test_inspect_has_required_vectors(self, engine):
        engine.create_collection("vec_check")
        schema = engine.inspect_collection("vec_check")
        vec_names = [v["name"] for v in schema["vectors"]]
        assert "text_embedding" in vec_names
        assert "image_embedding" in vec_names
        assert "sparse_embedding" in vec_names


class TestCollectionOptimize:
    def test_optimize_does_not_raise(self, populated_collection):
        # If engine returns None silently, that's fine
        populated_collection.optimize_collection("test")


class TestCollectionListFiles:
    def test_list_files_empty(self, collection):
        files = collection.list_files_in_collection("test")
        assert isinstance(files, list)

    def test_list_files_after_ingest(self, tmp_path, collection):
        f = tmp_path / "readme.md"
        f.write_text(MARKDOWN_SAMPLE)
        collection.ingest_file("test", f)
        files = collection.list_files_in_collection("test")
        assert len(files) >= 1
        sources = [f["source"] for f in files]
        assert "zrag://test/readme.md" in sources

    def test_list_files_has_chunk_count(self, tmp_path, collection):
        f = tmp_path / "doc.md"
        f.write_text(MARKDOWN_SAMPLE)
        collection.ingest_file("test", f)
        files = collection.list_files_in_collection("test")
        for file_info in files:
            assert "chunk_count" in file_info
            assert file_info["chunk_count"] > 0

    def test_list_files_filter(self, tmp_path, collection):
        py_file = tmp_path / "code.py"
        py_file.write_text(PYTHON_SAMPLE)
        md_file = tmp_path / "doc.md"
        md_file.write_text(MARKDOWN_SAMPLE)
        collection.ingest_file("test", py_file)
        collection.ingest_file("test", md_file)

        py_files = collection.list_files_in_collection("test", filter_pattern="*.py")
        assert all(f["source"].endswith(".py") for f in py_files)
