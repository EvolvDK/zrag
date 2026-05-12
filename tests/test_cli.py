"""
CLI end-to-end tests.

Uses Click's CliRunner for isolation (no actual daemon port needed).
Tests HTTP calls are intercepted via httpx mock — verifies CLI flag
parsing, output formatting, and command routing.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from zrag.cli import cli
from conftest import MARKDOWN_SAMPLE, PYTHON_SAMPLE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def runner():
    return CliRunner()


def _make_daemon_client_mock(responses: dict):
    """
    Returns a mock DaemonClient context manager that returns
    pre-baked responses for given endpoints.
    """
    mock_client = MagicMock()

    def get(endpoint, **kwargs):
        return responses.get(("GET", endpoint), {})

    def post(endpoint, **kwargs):
        return responses.get(("POST", endpoint), {})

    def delete(endpoint, **kwargs):
        return responses.get(("DELETE", endpoint), {})

    mock_client.__enter__ = lambda s: mock_client
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get = get
    mock_client.post = post
    mock_client.delete = delete
    return mock_client


# ---------------------------------------------------------------------------
# Collection commands
# ---------------------------------------------------------------------------

class TestCollectionCLI:
    def test_collection_list_empty(self, runner):
        mock = _make_daemon_client_mock({("GET", "/collections"): []})
        with patch("zrag.cli.ensure_daemon", return_value=mock):
            result = runner.invoke(cli, ["collection", "list"])
        assert result.exit_code == 0
        assert "No collections" in result.output

    def test_collection_list_with_data(self, runner):
        collections = [
            {"name": "docs", "document_count": 42, "size_bytes": 1024, "description": "Docs"},
        ]
        mock = _make_daemon_client_mock({("GET", "/collections"): collections})
        with patch("zrag.cli.ensure_daemon", return_value=mock):
            result = runner.invoke(cli, ["collection", "list"])
        assert result.exit_code == 0
        assert "docs" in result.output
        assert "42" in result.output

    def test_collection_add(self, runner, tmp_path):
        repo = tmp_path / "myrepo"
        repo.mkdir()
        (repo / "guide.md").write_text(MARKDOWN_SAMPLE)

        mock_response = {
            "name": "myrepo",
            "path": str(repo),
            "document_count": 3,
            "size_bytes": 2048,
        }
        mock = _make_daemon_client_mock({("POST", "/collections"): mock_response})
        with patch("zrag.cli.ensure_daemon", return_value=mock):
            result = runner.invoke(cli, ["collection", "add", str(repo)])
        assert result.exit_code == 0
        assert "myrepo" in result.output

    def test_collection_add_custom_name(self, runner, tmp_path):
        repo = tmp_path / "myrepo"
        repo.mkdir()
        mock_response = {"name": "custom", "path": str(repo), "document_count": 0, "size_bytes": 0}
        mock = _make_daemon_client_mock({("POST", "/collections"): mock_response})
        with patch("zrag.cli.ensure_daemon", return_value=mock):
            result = runner.invoke(cli, ["collection", "add", str(repo), "--name", "custom"])
        assert result.exit_code == 0
        assert "custom" in result.output

    def test_collection_remove_with_force(self, runner):
        mock = _make_daemon_client_mock({("DELETE", "/collections/testcol"): {"message": "removed"}})
        with patch("zrag.cli.ensure_daemon", return_value=mock):
            result = runner.invoke(cli, ["collection", "remove", "testcol", "--force"])
        assert result.exit_code == 0
        assert "removed" in result.output.lower() or "testcol" in result.output

    def test_collection_rename(self, runner):
        mock = _make_daemon_client_mock({
            ("POST", "/collections/rename"): {"name": "new_name", "path": "/some/path"}
        })
        with patch("zrag.cli.ensure_daemon", return_value=mock):
            result = runner.invoke(cli, ["collection", "rename", "old_name", "new_name"])
        assert result.exit_code == 0
        assert "new_name" in result.output

    def test_collection_inspect(self, runner):
        schema = {
            "name": "mydb",
            "fields": [{"name": "content", "data_type": "STRING"}],
            "vectors": [{"name": "text_embedding", "data_type": "VECTOR_FP32", "dimension": 384, "index_type": "HnswIndexParam"}],
        }
        mock = _make_daemon_client_mock({("GET", "/collections/mydb/inspect"): schema})
        with patch("zrag.cli.ensure_daemon", return_value=mock):
            result = runner.invoke(cli, ["collection", "inspect", "mydb"])
        assert result.exit_code == 0
        assert "content" in result.output
        assert "text_embedding" in result.output

    def test_collection_ls(self, runner):
        files = [
            {"source": "/docs/auth.md", "file_type": "markdown", "chunk_count": 5},
            {"source": "/src/auth.py", "file_type": "code_python", "chunk_count": 3},
        ]
        mock = _make_daemon_client_mock({("GET", "/collections/demo/files"): files})
        with patch("zrag.cli.ensure_daemon", return_value=mock):
            result = runner.invoke(cli, ["collection", "ls", "demo"])
        assert result.exit_code == 0
        assert "auth.md" in result.output
        assert "auth.py" in result.output

    def test_collection_optimize(self, runner):
        mock = _make_daemon_client_mock({("POST", "/collections/demo/optimize"): {"message": "done"}})
        with patch("zrag.cli.ensure_daemon", return_value=mock):
            result = runner.invoke(cli, ["collection", "optimize", "demo"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Search commands
# ---------------------------------------------------------------------------

SEARCH_RESPONSE = {
    "results": [
        {
            "id": "doc_abc_0",
            "score": 0.876,
            "fields": {
                "content": "JWT tokens are used for authentication.",
                "source": "/docs/auth.md",
                "context": "Auth documentation",
            },
        }
    ],
    "elapsed": 0.042,
    "explain": None,
}


class TestSearchCLI:
    def test_search_bm25(self, runner):
        mock = _make_daemon_client_mock({("POST", "/search/bm25"): SEARCH_RESPONSE})
        with patch("zrag.cli.ensure_daemon", return_value=mock):
            result = runner.invoke(cli, ["search", "JWT authentication"])
        assert result.exit_code == 0
        assert "JWT" in result.output

    def test_vsearch(self, runner):
        mock = _make_daemon_client_mock({("POST", "/search/vector"): SEARCH_RESPONSE})
        with patch("zrag.cli.ensure_daemon", return_value=mock):
            result = runner.invoke(cli, ["vsearch", "authentication token"])
        assert result.exit_code == 0

    def test_query_hybrid(self, runner):
        mock = _make_daemon_client_mock({("POST", "/search/hybrid"): SEARCH_RESPONSE})
        with patch("zrag.cli.ensure_daemon", return_value=mock):
            result = runner.invoke(cli, ["query", "how does login work"])
        assert result.exit_code == 0

    def test_search_json_output(self, runner):
        mock = _make_daemon_client_mock({("POST", "/search/bm25"): SEARCH_RESPONSE})
        with patch("zrag.cli.ensure_daemon", return_value=mock):
            result = runner.invoke(cli, ["search", "auth", "--json"])
        assert result.exit_code == 0
        # JSON output should be parseable
        parsed = json.loads(result.output.strip())
        assert isinstance(parsed, list)

    def test_search_csv_output(self, runner):
        mock = _make_daemon_client_mock({("POST", "/search/bm25"): SEARCH_RESPONSE})
        with patch("zrag.cli.ensure_daemon", return_value=mock):
            result = runner.invoke(cli, ["search", "auth", "--csv"])
        assert result.exit_code == 0
        assert "id" in result.output  # CSV header

    def test_search_md_output(self, runner):
        mock = _make_daemon_client_mock({("POST", "/search/bm25"): SEARCH_RESPONSE})
        with patch("zrag.cli.ensure_daemon", return_value=mock):
            result = runner.invoke(cli, ["search", "auth", "--md"])
        assert result.exit_code == 0
        assert "# Search Results" in result.output

    def test_search_xml_output(self, runner):
        from xml.etree import ElementTree as ET
        mock = _make_daemon_client_mock({("POST", "/search/bm25"): SEARCH_RESPONSE})
        with patch("zrag.cli.ensure_daemon", return_value=mock):
            result = runner.invoke(cli, ["search", "auth", "--xml"])
        assert result.exit_code == 0
        ET.fromstring(result.output.strip())  # must be valid XML

    def test_search_top_k_flag(self, runner):
        mock = _make_daemon_client_mock({("POST", "/search/bm25"): SEARCH_RESPONSE})
        with patch("zrag.cli.ensure_daemon", return_value=mock):
            result = runner.invoke(cli, ["search", "auth", "-n", "3"])
        assert result.exit_code == 0

    def test_search_threshold_filters(self, runner):
        low_score_response = {
            "results": [{"id": "x", "score": 0.1, "fields": {"content": "c", "source": "s.md", "context": ""}}],
            "elapsed": 0.01,
        }
        mock = _make_daemon_client_mock({("POST", "/search/bm25"): low_score_response})
        with patch("zrag.cli.ensure_daemon", return_value=mock):
            result = runner.invoke(cli, ["search", "auth", "--threshold", "0.5"])
        assert result.exit_code == 0
        assert "No results" in result.output

    def test_search_collection_flag(self, runner):
        mock = _make_daemon_client_mock({("POST", "/search/bm25"): SEARCH_RESPONSE})
        with patch("zrag.cli.ensure_daemon", return_value=mock):
            result = runner.invoke(cli, ["search", "auth", "-c", "mycollection"])
        assert result.exit_code == 0

    def test_query_no_expansion_flag(self, runner):
        mock = _make_daemon_client_mock({("POST", "/search/hybrid"): SEARCH_RESPONSE})
        with patch("zrag.cli.ensure_daemon", return_value=mock):
            result = runner.invoke(cli, ["query", "auth", "--no-expansion"])
        assert result.exit_code == 0

    def test_query_explain_flag(self, runner):
        explained = {**SEARCH_RESPONSE, "explain": {
            "lex_queries": ["auth"], "vec_queries": ["auth"],
            "hyde": "", "documents": []
        }}
        mock = _make_daemon_client_mock({("POST", "/search/hybrid"): explained})
        with patch("zrag.cli.ensure_daemon", return_value=mock):
            result = runner.invoke(cli, ["query", "auth", "--explain"])
        assert result.exit_code == 0

    def test_search_no_results(self, runner):
        mock = _make_daemon_client_mock({("POST", "/search/bm25"): {"results": [], "elapsed": 0.01}})
        with patch("zrag.cli.ensure_daemon", return_value=mock):
            result = runner.invoke(cli, ["search", "xyzzy_quux"])
        assert result.exit_code == 0
        assert "No results" in result.output

    def test_search_full_flag(self, runner):
        mock = _make_daemon_client_mock({("POST", "/search/bm25"): SEARCH_RESPONSE})
        with patch("zrag.cli.ensure_daemon", return_value=mock):
            result = runner.invoke(cli, ["search", "auth", "--full"])
        assert result.exit_code == 0
        assert "JWT tokens are used" in result.output

    def test_search_files_flag(self, runner):
        mock = _make_daemon_client_mock({("POST", "/search/bm25"): SEARCH_RESPONSE})
        with patch("zrag.cli.ensure_daemon", return_value=mock):
            result = runner.invoke(cli, ["search", "auth", "--files"])
        assert result.exit_code == 0
        # Files format shows source column
        assert "auth.md" in result.output


# ---------------------------------------------------------------------------
# Ingest commands
# ---------------------------------------------------------------------------

class TestIngestCLI:
    def test_ingest_file(self, runner, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text(MARKDOWN_SAMPLE)
        mock = _make_daemon_client_mock({("POST", "/ingest/file"): {"added": 3, "elapsed": 0.1, "errors": []}})
        with patch("zrag.cli.ensure_daemon", return_value=mock):
            result = runner.invoke(cli, ["ingest", "file", "demo", str(f)])
        assert result.exit_code == 0
        assert "ingested" in result.output.lower()

    def test_ingest_url(self, runner):
        mock = _make_daemon_client_mock({("POST", "/ingest/url"): {"added": 2, "elapsed": 0.2, "errors": []}})
        with patch("zrag.cli.ensure_daemon", return_value=mock):
            result = runner.invoke(cli, ["ingest", "url", "demo", "https://example.com/auth"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Context commands
# ---------------------------------------------------------------------------

class TestContextCLI:
    def test_context_add(self, runner):
        mock = _make_daemon_client_mock({
            ("POST", "/context/add"): {"path": "/docs", "description": "Docs", "created_at": 0, "updated_at": 0}
        })
        with patch("zrag.cli.ensure_daemon", return_value=mock):
            result = runner.invoke(cli, ["context", "add", "/docs", "Docs folder"])
        assert result.exit_code == 0

    def test_context_list(self, runner):
        contexts = [{"path": "/docs", "description": "Docs"}, {"path": "global", "description": "Global"}]
        mock = _make_daemon_client_mock({("GET", "/context/list"): contexts})
        with patch("zrag.cli.ensure_daemon", return_value=mock):
            result = runner.invoke(cli, ["context", "list"])
        assert result.exit_code == 0
        assert "/docs" in result.output

    def test_context_remove(self, runner):
        mock = _make_daemon_client_mock({("DELETE", "/context/remove?path=/docs"): {"message": "removed"}})
        with patch("zrag.cli.ensure_daemon", return_value=mock):
            result = runner.invoke(cli, ["context", "remove", "/docs"])
        assert result.exit_code == 0

    def test_context_check(self, runner):
        check_response = {"missing_context": ["/docs/auth.md", "/src/auth.py"]}
        mock = _make_daemon_client_mock({("GET", "/context/check/demo"): check_response})
        with patch("zrag.cli.ensure_daemon", return_value=mock):
            result = runner.invoke(cli, ["context", "check", "demo"])
        assert result.exit_code == 0
        assert "auth.md" in result.output

    def test_context_check_all_covered(self, runner):
        mock = _make_daemon_client_mock({("GET", "/context/check/demo"): {"missing_context": []}})
        with patch("zrag.cli.ensure_daemon", return_value=mock):
            result = runner.invoke(cli, ["context", "check", "demo"])
        assert result.exit_code == 0
        assert "All sources have context" in result.output


# ---------------------------------------------------------------------------
# Update & embed commands
# ---------------------------------------------------------------------------

class TestOrchestrationCLI:
    def test_update_command(self, runner):
        update_response = {"added": 5, "updated": 1, "removed": 0, "elapsed": 0.5, "errors": []}
        mock = _make_daemon_client_mock({("POST", "/collections/demo/update"): update_response})
        with patch("zrag.cli.ensure_daemon", return_value=mock):
            result = runner.invoke(cli, ["update", "demo"])
        assert result.exit_code == 0
        assert "5" in result.output

    def test_update_with_pull_flag(self, runner):
        mock = _make_daemon_client_mock({("POST", "/collections/demo/update"): {"added": 0, "updated": 0, "removed": 0, "elapsed": 0.1, "errors": []}})
        with patch("zrag.cli.ensure_daemon", return_value=mock):
            result = runner.invoke(cli, ["update", "demo", "--pull"])
        assert result.exit_code == 0

    def test_embed_command(self, runner):
        mock = _make_daemon_client_mock({("POST", "/collections/demo/embed"): {"added": 10, "elapsed": 1.2}})
        with patch("zrag.cli.ensure_daemon", return_value=mock):
            result = runner.invoke(cli, ["embed", "demo"])
        assert result.exit_code == 0

    def test_status_command(self, runner):
        status = {"daemon_running": True, "collections_loaded": 2, "models_loaded": True}
        mock = _make_daemon_client_mock({("GET", "/status"): status})
        with patch("zrag.cli.ensure_daemon", return_value=mock):
            result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0
        assert "Running" in result.output


# ---------------------------------------------------------------------------
# Version flag
# ---------------------------------------------------------------------------

class TestCLIVersion:
    def test_version_flag(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output
