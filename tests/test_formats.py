"""
Output format tests.

Covers: JSON, CSV, XML, Markdown, CLI — structural correctness,
field presence, encoding, edge cases (empty results, special chars).
"""

import csv
import json
import io
from xml.etree import ElementTree as ET

import pytest

from zrag.formatters import OutputFormat, OutputFormatter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RESULTS = [
    {
        "id": "doc_abc123_0",
        "score": 0.8754,
        "source": "/home/user/docs/auth.md",
        "content": "JWT tokens expire after 24 hours.",
        "chunk_id": 0,
        "file_type": "markdown",
        "context": "Auth module documentation",
    },
    {
        "id": "doc_def456_1",
        "score": 0.6231,
        "source": "/home/user/src/auth.py",
        "content": "def verify_token(token: str) -> bool:\n    return token in sessions",
        "chunk_id": 1,
        "file_type": "code_python",
        "context": None,
    },
]

EMPTY_RESULTS = []

SPECIAL_CHAR_RESULTS = [
    {
        "id": "doc_special_0",
        "score": 0.5,
        "source": "/path/to/O'Brien & Smith/file.md",
        "content": 'Content with <special> "chars" & symbols.',
        "chunk_id": 0,
        "file_type": "markdown",
        "context": None,
    }
]


@pytest.fixture
def formatter():
    return OutputFormatter()


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

class TestJSONFormat:
    def test_valid_json_output(self, formatter):
        output = formatter.format(SAMPLE_RESULTS, format=OutputFormat.JSON)
        parsed = json.loads(output)  # must not raise
        assert isinstance(parsed, list)

    def test_json_preserves_all_results(self, formatter):
        output = formatter.format(SAMPLE_RESULTS, format=OutputFormat.JSON)
        parsed = json.loads(output)
        assert len(parsed) == len(SAMPLE_RESULTS)

    def test_json_has_required_fields(self, formatter):
        output = formatter.format(SAMPLE_RESULTS, format=OutputFormat.JSON)
        parsed = json.loads(output)
        for item in parsed:
            assert "id" in item
            assert "score" in item
            assert "source" in item
            assert "content" in item

    def test_json_score_is_float(self, formatter):
        output = formatter.format(SAMPLE_RESULTS, format=OutputFormat.JSON)
        parsed = json.loads(output)
        for item in parsed:
            assert isinstance(item["score"], float)

    def test_json_empty_results(self, formatter):
        output = formatter.format(EMPTY_RESULTS, format=OutputFormat.JSON)
        parsed = json.loads(output)
        assert parsed == []

    def test_json_special_chars_safe(self, formatter):
        output = formatter.format(SPECIAL_CHAR_RESULTS, format=OutputFormat.JSON)
        parsed = json.loads(output)
        assert "O'Brien" in parsed[0]["source"]
        assert "<special>" in parsed[0]["content"]

    def test_json_pretty_indent(self, formatter):
        output = formatter.format(SAMPLE_RESULTS, format=OutputFormat.JSON, indent=2)
        assert "\n" in output  # indented = multiline


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

class TestCSVFormat:
    def test_valid_csv_output(self, formatter):
        output = formatter.format(SAMPLE_RESULTS, format=OutputFormat.CSV)
        reader = csv.DictReader(io.StringIO(output))
        rows = list(reader)  # must not raise
        assert len(rows) == len(SAMPLE_RESULTS)

    def test_csv_has_header(self, formatter):
        output = formatter.format(SAMPLE_RESULTS, format=OutputFormat.CSV)
        first_line = output.split("\n")[0]
        assert "id" in first_line
        assert "source" in first_line

    def test_csv_has_correct_row_count(self, formatter):
        output = formatter.format(SAMPLE_RESULTS, format=OutputFormat.CSV)
        reader = csv.DictReader(io.StringIO(output))
        rows = list(reader)
        assert len(rows) == 2

    def test_csv_empty_results(self, formatter):
        output = formatter.format(EMPTY_RESULTS, format=OutputFormat.CSV)
        assert output == ""

    def test_csv_score_present(self, formatter):
        output = formatter.format(SAMPLE_RESULTS, format=OutputFormat.CSV)
        reader = csv.DictReader(io.StringIO(output))
        rows = list(reader)
        assert rows[0]["score"] == "0.8754"

    def test_csv_custom_fields(self, formatter):
        output = formatter.format(
            SAMPLE_RESULTS, format=OutputFormat.CSV, fields=["id", "score", "content"]
        )
        reader = csv.DictReader(io.StringIO(output))
        rows = list(reader)
        assert "id" in rows[0]
        assert "content" in rows[0]
        assert "source" not in rows[0]


# ---------------------------------------------------------------------------
# XML
# ---------------------------------------------------------------------------

class TestXMLFormat:
    def test_valid_xml_output(self, formatter):
        output = formatter.format(SAMPLE_RESULTS, format=OutputFormat.XML)
        ET.fromstring(output)  # must not raise ParseError

    def test_xml_root_is_results(self, formatter):
        output = formatter.format(SAMPLE_RESULTS, format=OutputFormat.XML)
        root = ET.fromstring(output)
        assert root.tag == "results"

    def test_xml_document_count(self, formatter):
        output = formatter.format(SAMPLE_RESULTS, format=OutputFormat.XML)
        root = ET.fromstring(output)
        docs = root.findall("document")
        assert len(docs) == len(SAMPLE_RESULTS)

    def test_xml_has_id_attribute(self, formatter):
        output = formatter.format(SAMPLE_RESULTS, format=OutputFormat.XML)
        root = ET.fromstring(output)
        for doc in root.findall("document"):
            assert doc.get("id") is not None

    def test_xml_has_score_attribute(self, formatter):
        output = formatter.format(SAMPLE_RESULTS, format=OutputFormat.XML)
        root = ET.fromstring(output)
        for doc in root.findall("document"):
            assert doc.get("score") is not None

    def test_xml_has_fields_element(self, formatter):
        output = formatter.format(SAMPLE_RESULTS, format=OutputFormat.XML)
        root = ET.fromstring(output)
        for doc in root.findall("document"):
            assert doc.find("fields") is not None

    def test_xml_empty_results(self, formatter):
        output = formatter.format(EMPTY_RESULTS, format=OutputFormat.XML)
        root = ET.fromstring(output)
        assert len(root.findall("document")) == 0

    def test_xml_special_chars_escaped(self, formatter):
        output = formatter.format(SPECIAL_CHAR_RESULTS, format=OutputFormat.XML)
        ET.fromstring(output)  # valid XML = properly escaped
        # Content with <special> must be in text, not break XML
        assert "special" in output


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

class TestMarkdownFormat:
    def test_markdown_output_is_string(self, formatter):
        output = formatter.format(SAMPLE_RESULTS, format=OutputFormat.MARKDOWN)
        assert isinstance(output, str)

    def test_markdown_has_h1_header(self, formatter):
        output = formatter.format(SAMPLE_RESULTS, format=OutputFormat.MARKDOWN)
        assert output.startswith("# Search Results")

    def test_markdown_has_numbered_sections(self, formatter):
        output = formatter.format(SAMPLE_RESULTS, format=OutputFormat.MARKDOWN)
        assert "## 1." in output
        assert "## 2." in output

    def test_markdown_has_score(self, formatter):
        output = formatter.format(SAMPLE_RESULTS, format=OutputFormat.MARKDOWN)
        assert "0.8754" in output

    def test_markdown_has_source(self, formatter):
        output = formatter.format(SAMPLE_RESULTS, format=OutputFormat.MARKDOWN)
        assert "auth.md" in output

    def test_markdown_has_content_preview(self, formatter):
        output = formatter.format(SAMPLE_RESULTS, format=OutputFormat.MARKDOWN)
        assert "JWT" in output

    def test_markdown_empty_results(self, formatter):
        output = formatter.format(EMPTY_RESULTS, format=OutputFormat.MARKDOWN)
        assert "No results" in output

    def test_markdown_context_included(self, formatter):
        output = formatter.format(SAMPLE_RESULTS, format=OutputFormat.MARKDOWN)
        assert "Auth module documentation" in output

    def test_markdown_separator_between_results(self, formatter):
        output = formatter.format(SAMPLE_RESULTS, format=OutputFormat.MARKDOWN)
        assert "---" in output

    def test_markdown_preview_truncated(self, formatter):
        long_result = [{
            "id": "x",
            "score": 0.9,
            "source": "a.md",
            "content": "A" * 500,
            "context": None,
        }]
        output = formatter.format(long_result, format=OutputFormat.MARKDOWN, max_preview=200)
        assert "..." in output


# ---------------------------------------------------------------------------
# format_get_result
# ---------------------------------------------------------------------------

class TestFormatGetResult:
    SINGLE = {
        "id": "doc_xyz_0",
        "score": 0.95,
        "source": "/docs/auth.md",
        "content": "Authentication guide content.",
        "file_type": "markdown",
        "context": "Auth docs",
        "line_range": "10-20",
    }

    def test_json_single(self, formatter):
        output = formatter.format_get_result(self.SINGLE, format=OutputFormat.JSON)
        parsed = json.loads(output)
        assert parsed["id"] == "doc_xyz_0"

    def test_cli_single_contains_id(self, formatter):
        output = formatter.format_get_result(self.SINGLE, format=OutputFormat.CLI)
        assert "doc_xyz_0" in output

    def test_cli_single_contains_line_range(self, formatter):
        output = formatter.format_get_result(self.SINGLE, format=OutputFormat.CLI)
        assert "10-20" in output

    def test_cli_single_contains_source(self, formatter):
        output = formatter.format_get_result(self.SINGLE, format=OutputFormat.CLI)
        assert "auth.md" in output

    def test_cli_single_contains_content(self, formatter):
        output = formatter.format_get_result(self.SINGLE, format=OutputFormat.CLI)
        assert "Authentication guide" in output


# ---------------------------------------------------------------------------
# Unknown format raises
# ---------------------------------------------------------------------------

class TestUnknownFormat:
    def test_unknown_format_raises_value_error(self, formatter):
        with pytest.raises((ValueError, KeyError, AttributeError)):
            formatter.format(SAMPLE_RESULTS, format="totally_unknown")
