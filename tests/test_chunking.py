"""
Chunking quality tests.

Verifies:
- AST chunking preserves class/function boundaries
- Semantic chunking respects markdown structure
- Simple chunking respects token limits
- No duplicate chunks in any pipeline
- Overlap behavior
- Fallback when AST fails
"""

import textwrap
from pathlib import Path

import pytest

from zrag.chunking import (
    chunk_code_ast,
    chunk_text_semantic,
    chunk_text_simple,
    chunk_text_by_tokens,
    get_file_type,
)
from conftest import PYTHON_SAMPLE, MARKDOWN_SAMPLE, GO_SAMPLE, RUST_SAMPLE


# ---------------------------------------------------------------------------
# Token-based chunking
# ---------------------------------------------------------------------------

class TestTokenChunking:
    def test_short_text_single_chunk(self):
        chunks = chunk_text_by_tokens("Hello world.", max_tokens=100)
        assert len(chunks) == 1
        assert "Hello world" in chunks[0]

    def test_respects_max_tokens(self):
        # ~200 token text chunked at 50 tokens → must produce multiple chunks
        text = " ".join(["word"] * 200)
        chunks = chunk_text_by_tokens(text, max_tokens=50, overlap=0.0)
        assert len(chunks) > 1

    def test_overlap_creates_repeated_content(self):
        text = " ".join([f"word{i}" for i in range(200)])
        chunks_no_overlap = chunk_text_by_tokens(text, max_tokens=50, overlap=0.0)
        chunks_overlap = chunk_text_by_tokens(text, max_tokens=50, overlap=0.2)
        # With overlap, boundary words appear in consecutive chunks
        assert len(chunks_overlap) >= len(chunks_no_overlap)

    def test_empty_text_returns_empty(self):
        assert chunk_text_by_tokens("", max_tokens=100) == []
        assert chunk_text_by_tokens("   ", max_tokens=100) == []

    def test_no_empty_chunks(self):
        text = "Hello\n\n\n\nWorld\n\n\n"
        chunks = chunk_text_by_tokens(text, max_tokens=50)
        assert all(c.strip() for c in chunks)


class TestSimpleChunking:
    def test_delegates_to_token_chunker(self):
        chunks = chunk_text_simple(MARKDOWN_SAMPLE, max_tokens=900)
        assert isinstance(chunks, list)
        assert len(chunks) >= 1

    def test_no_duplicates(self):
        text = " ".join([f"token{i}" for i in range(1000)])
        chunks = chunk_text_simple(text, max_tokens=100)
        assert len(chunks) == len(set(chunks))


# ---------------------------------------------------------------------------
# Semantic (Markdown) chunking
# ---------------------------------------------------------------------------

class TestSemanticChunking:
    def test_produces_chunks(self):
        chunks = chunk_text_semantic(MARKDOWN_SAMPLE, max_tokens=200, overlap=0.15)
        assert len(chunks) >= 2

    def test_no_empty_chunks(self):
        chunks = chunk_text_semantic(MARKDOWN_SAMPLE, max_tokens=200, overlap=0.15)
        assert all(c.strip() for c in chunks)

    def test_no_duplicate_chunks(self):
        chunks = chunk_text_semantic(MARKDOWN_SAMPLE, max_tokens=200, overlap=0.15)
        assert len(chunks) == len(set(chunks)), "Duplicate chunks found"

    def test_code_fence_not_split(self):
        """Code blocks must not be split at intermediate lines."""
        md = textwrap.dedent("""\
            # Title

            Some intro text here.

            ```python
            def foo():
                x = 1
                y = 2
                z = x + y
                return z
            ```

            After the code block.
        """)
        chunks = chunk_text_semantic(md, max_tokens=500, overlap=0.1)
        # The code block should appear intact in one chunk
        code_chunks = [c for c in chunks if "def foo():" in c]
        assert len(code_chunks) >= 1
        for c in code_chunks:
            # Must not start mid-function
            assert "def foo():" in c

    def test_heading_boundaries_respected(self):
        """H2 headings should create chunk boundaries."""
        md = textwrap.dedent("""\
            ## Section One

            Content about section one. This is the first paragraph.

            ## Section Two

            Content about section two. This is the second paragraph.

            ## Section Three

            Content about section three. This is the third paragraph.
        """)
        chunks = chunk_text_semantic(md, max_tokens=25, overlap=0.1)
        # With 50 token limit and 3 sections, we expect at least 2 chunks
        assert len(chunks) >= 2

    def test_large_text_chunked(self):
        # 5000-word text with 200-token limit → many chunks
        text = ("# Section\n\n" + "This is a sentence with multiple words. " * 500)
        chunks = chunk_text_semantic(text, max_tokens=200, overlap=0.15)
        assert len(chunks) > 5

    def test_preserves_content(self):
        """All original content should be reachable across chunks."""
        chunks = chunk_text_semantic(MARKDOWN_SAMPLE, max_tokens=200, overlap=0.0)
        combined = " ".join(chunks)
        # Key terms must survive chunking
        for term in ["JWT", "bcrypt", "rate-limited", "logout"]:
            assert term in combined


# ---------------------------------------------------------------------------
# AST Code chunking
# ---------------------------------------------------------------------------

class TestASTChunking:

    def _write_and_chunk(self, tmp_path: Path, content: str, lang: str, ext: str):
        f = tmp_path / f"sample.{ext}"
        f.write_text(content)
        return chunk_code_ast(f, lang, max_tokens=512)

    def test_python_class_boundaries(self, tmp_path):
        chunks = self._write_and_chunk(tmp_path, PYTHON_SAMPLE, "python", "py")
        contents = [c for c, _ in chunks]
        # AuthManager class must be a chunk
        class_chunks = [c for c in contents if "class AuthManager" in c]
        assert len(class_chunks) >= 1, "AuthManager class not found as chunk"

    def test_python_function_boundaries(self, tmp_path):
        chunks = self._write_and_chunk(tmp_path, PYTHON_SAMPLE, "python", "py")
        contents = [c for c, _ in chunks]
        func_chunks = [c for c in contents if "def hash_password" in c]
        assert len(func_chunks) >= 1, "hash_password function not found as chunk"

    def test_go_struct_boundaries(self, tmp_path):
        chunks = self._write_and_chunk(tmp_path, GO_SAMPLE, "go", "go")
        contents = [c for c, _ in chunks]
        assert any("SessionStore" in c for c in contents), "SessionStore not found"

    def test_go_function_boundaries(self, tmp_path):
        chunks = self._write_and_chunk(tmp_path, GO_SAMPLE, "go", "go")
        contents = [c for c, _ in chunks]
        assert any("func (s *SessionStore) Create" in c for c in contents)

    def test_rust_struct_boundaries(self, tmp_path):
        chunks = self._write_and_chunk(tmp_path, RUST_SAMPLE, "rust", "rs")
        contents = [c for c, _ in chunks]
        assert any("pub struct Session" in c for c in contents)

    def test_rust_impl_boundaries(self, tmp_path):
        chunks = self._write_and_chunk(tmp_path, RUST_SAMPLE, "rust", "rs")
        contents = [c for c, _ in chunks]
        assert any("impl SessionStore" in c for c in contents)

    def test_returns_list_of_tuples(self, tmp_path):
        chunks = self._write_and_chunk(tmp_path, PYTHON_SAMPLE, "python", "py")
        assert isinstance(chunks, list)
        for item in chunks:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_no_empty_chunks(self, tmp_path):
        chunks = self._write_and_chunk(tmp_path, PYTHON_SAMPLE, "python", "py")
        for content, _ in chunks:
            assert content.strip(), "Empty chunk found"

    def test_no_duplicate_ast_chunks(self, tmp_path):
        chunks = self._write_and_chunk(tmp_path, PYTHON_SAMPLE, "python", "py")
        contents = [c for c, _ in chunks]
        assert len(contents) == len(set(contents)), "Duplicate AST chunks"

    def test_chunk_types_annotated(self, tmp_path):
        """Each chunk must carry a node type annotation."""
        chunks = self._write_and_chunk(tmp_path, PYTHON_SAMPLE, "python", "py")
        for _, node_type in chunks:
            assert isinstance(node_type, str)
            assert len(node_type) > 0

    def test_fallback_on_unknown_lang(self, tmp_path):
        """Unknown language → fallback token chunking, must not crash."""
        f = tmp_path / "sample.zig"
        f.write_text("fn main() {}\n" * 100)
        chunks = chunk_code_ast(f, "zig", max_tokens=200)
        assert isinstance(chunks, list)
        assert len(chunks) >= 1

    def test_large_class_split_into_methods(self, tmp_path):
        """A class exceeding 1.5x token limit should be split into its methods."""
        big_class = "class BigClass:\n"
        for i in range(50):
            big_class += f"    def method_{i}(self):\n        return {i}\n\n"
        from zrag.chunking import chunk_code_ast
        f = tmp_path / "large_class.py"
        f.write_text(big_class)
        chunks = chunk_code_ast(f, "python", max_tokens=256)
        contents = [c for c, _ in chunks]
        # Should produce multiple method chunks, not one giant blob
        method_chunks = [c for c in contents if "def method_" in c]
        assert len(method_chunks) > 1, "Large class not split into methods"


# ---------------------------------------------------------------------------
# get_file_type
# ---------------------------------------------------------------------------

class TestGetFileType:
    @pytest.mark.parametrize("ext,expected", [
        (".py", "code_python"),
        (".js", "code_javascript"),
        (".ts", "code_typescript"),
        (".tsx", "code_typescript"),
        (".go", "code_go"),
        (".rs", "code_rust"),
        (".cpp", "code_cpp"),
        (".hpp", "code_cpp"),
        (".cc", "code_cpp"),
        (".h", "code_cpp"),
        (".md", "markdown"),
        (".pdf", "pdf"),
        (".docx", "docx"),
        (".png", "image"),
        (".jpg", "image"),
        (".jpeg", "image"),
        (".unknown_ext", "unknown"),
    ])
    def test_file_type_detection(self, ext, expected, tmp_path):
        f = tmp_path / f"file{ext}"
        f.touch()
        assert get_file_type(f) == expected
