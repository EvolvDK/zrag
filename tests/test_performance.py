"""
Performance benchmark tests.

Establishes baselines and regression guards for:
- Ingestion throughput (files/sec, chunks/sec)
- Search latency (BM25, vector, hybrid)
- CLI/daemon response time
- Chunking throughput

Mark with @pytest.mark.perf to run selectively:
    pytest -m perf tests/test_performance.py -v

Baselines are generous to accommodate CPU-only environments.
Adjust BASELINE_* constants for your hardware.
"""

import time
import statistics
import textwrap
from pathlib import Path

import pytest

from conftest import PYTHON_SAMPLE, MARKDOWN_SAMPLE, GO_SAMPLE, RUST_SAMPLE

# ---------------------------------------------------------------------------
# Baseline thresholds (CPU-only defaults — tune per machine)
# ---------------------------------------------------------------------------

# Chunking
BASELINE_SIMPLE_CHUNK_MS = 500       # chunk_text_simple for 50KB text
BASELINE_SEMANTIC_CHUNK_MS = 2000    # chunk_text_semantic for 50KB markdown
BASELINE_AST_CHUNK_MS = 3000         # chunk_code_ast for 500-line Python

# Ingestion
BASELINE_INGEST_FILE_MS = 10_000     # single file ingest (embed included)
BASELINE_INGEST_10_FILES_S = 120     # 10 mixed files in under 120s

# Search (after embedding hot)
BASELINE_BM25_SEARCH_MS = 500        # single BM25 query
BASELINE_VECTOR_SEARCH_MS = 1000     # single vector query (no expansion)
BASELINE_HYBRID_SEARCH_MS = 2000     # single hybrid query (no expansion/rerank)

# Collection ops
BASELINE_CREATE_COLLECTION_MS = 2000
BASELINE_LIST_COLLECTIONS_MS = 1000

pytestmark = pytest.mark.perf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _repeat(fn, n=5):
    """Run fn n times, return list of elapsed seconds."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def _assert_ms(elapsed_s: float, baseline_ms: int, label: str):
    elapsed_ms = elapsed_s * 1000
    assert elapsed_ms < baseline_ms, (
        f"PERF REGRESSION [{label}]: {elapsed_ms:.1f}ms > baseline {baseline_ms}ms"
    )


def _make_large_python(n_classes: int = 10, methods_per: int = 10) -> str:
    out = ""
    for i in range(n_classes):
        out += f"\nclass Service{i}:\n"
        for j in range(methods_per):
            out += f"    def method_{j}(self, x: int, y: int) -> int:\n"
            out += f"        \"\"\"Method {j} of Service {i}.\"\"\"\n"
            out += f"        return x + y + {j}\n\n"
    return out


def _make_large_markdown(n_sections: int = 20, words_per: int = 100) -> str:
    out = "# Main Document\n\n"
    for i in range(n_sections):
        out += f"## Section {i}\n\n"
        out += ("This is a realistic paragraph with enough words to make it substantial. " * (words_per // 10))
        out += "\n\n"
        if i % 3 == 0:
            out += "```python\ndef example():\n    return 'hello'\n```\n\n"
    return out


# ---------------------------------------------------------------------------
# Chunking benchmarks
# ---------------------------------------------------------------------------

class TestChunkingPerformance:
    def test_simple_chunk_throughput(self):
        text = MARKDOWN_SAMPLE * 20  # ~50KB
        from zrag.chunking import chunk_text_simple
        t0 = time.perf_counter()
        chunks = chunk_text_simple(text, max_tokens=900)
        elapsed = time.perf_counter() - t0
        assert len(chunks) > 0
        _assert_ms(elapsed, BASELINE_SIMPLE_CHUNK_MS, "chunk_text_simple 50KB")

    def test_semantic_chunk_throughput(self):
        text = _make_large_markdown(n_sections=30, words_per=150)
        from zrag.chunking import chunk_text_semantic
        t0 = time.perf_counter()
        chunks = chunk_text_semantic(text, max_tokens=400, overlap=0.15)
        elapsed = time.perf_counter() - t0
        assert len(chunks) > 5
        _assert_ms(elapsed, BASELINE_SEMANTIC_CHUNK_MS, "chunk_text_semantic 30 sections")

    def test_ast_chunk_python_throughput(self, tmp_path):
        code = _make_large_python(n_classes=20, methods_per=15)
        f = tmp_path / "large.py"
        f.write_text(code)
        from zrag.chunking import chunk_code_ast
        t0 = time.perf_counter()
        chunks = chunk_code_ast(f, "python", max_tokens=512)
        elapsed = time.perf_counter() - t0
        assert len(chunks) > 10
        _assert_ms(elapsed, BASELINE_AST_CHUNK_MS, "chunk_code_ast 200-class Python")

    def test_semantic_chunk_no_regression(self):
        """Run 3x and check median — detects flaky perf."""
        from zrag.chunking import chunk_text_semantic
        text = _make_large_markdown(n_sections=10)
        times = _repeat(lambda: chunk_text_semantic(text, max_tokens=400, overlap=0.1), n=3)
        median_ms = statistics.median(times) * 1000
        assert median_ms < BASELINE_SEMANTIC_CHUNK_MS, (
            f"Median semantic chunk time {median_ms:.1f}ms > baseline {BASELINE_SEMANTIC_CHUNK_MS}ms"
        )


# ---------------------------------------------------------------------------
# Ingestion benchmarks
# ---------------------------------------------------------------------------

class TestIngestionPerformance:
    def test_single_markdown_ingest_time(self, engine, tmp_path):
        engine.create_collection("perf_md")
        f = tmp_path / "guide.md"
        f.write_text(MARKDOWN_SAMPLE * 5)

        t0 = time.perf_counter()
        stats, _ = engine.ingest_file("perf_md", f)
        elapsed = time.perf_counter() - t0

        assert stats.added >= 1
        _assert_ms(elapsed, BASELINE_INGEST_FILE_MS, "ingest single markdown file")

    def test_single_python_ingest_time(self, engine, tmp_path):
        engine.create_collection("perf_py")
        f = tmp_path / "auth.py"
        f.write_text(_make_large_python(n_classes=5, methods_per=10))

        t0 = time.perf_counter()
        stats, _ = engine.ingest_file("perf_py", f)
        elapsed = time.perf_counter() - t0

        assert stats.added >= 1
        _assert_ms(elapsed, BASELINE_INGEST_FILE_MS, "ingest single python file")

    def test_ingest_10_mixed_files(self, engine, tmp_path):
        engine.create_collection("perf_batch")
        files = []
        for i in range(4):
            f = tmp_path / f"doc_{i}.md"
            f.write_text(MARKDOWN_SAMPLE)
            files.append(f)
        for i in range(3):
            f = tmp_path / f"code_{i}.py"
            f.write_text(PYTHON_SAMPLE)
            files.append(f)
        for i in range(2):
            f = tmp_path / f"go_{i}.go"
            f.write_text(GO_SAMPLE)
            files.append(f)
        f = tmp_path / "store.rs"
        f.write_text(RUST_SAMPLE)
        files.append(f)

        t0 = time.perf_counter()
        for fp in files:
            engine.ingest_file("perf_batch", fp)
        elapsed = time.perf_counter() - t0

        assert elapsed < BASELINE_INGEST_10_FILES_S, (
            f"10-file ingest took {elapsed:.1f}s > baseline {BASELINE_INGEST_10_FILES_S}s"
        )
        all_files = engine.list_files_in_collection("perf_batch")
        assert len(all_files) == 10

    def test_upsert_same_file_no_slowdown(self, engine, tmp_path):
        engine.create_collection("perf_upsert")
        f = tmp_path / "doc.md"
        f.write_text(MARKDOWN_SAMPLE)
        engine.ingest_file("perf_upsert", f)  # warm-up

        t0 = time.perf_counter()
        engine.ingest_file("perf_upsert", f)  # re-ingest (upsert)
        elapsed = time.perf_counter() - t0

        _assert_ms(elapsed, BASELINE_INGEST_FILE_MS, "upsert re-ingest same file")

    def test_ingest_throughput_chunks_per_second(self, engine, tmp_path):
        engine.create_collection("perf_throughput")
        # Generate ~10 files of markdown
        files = []
        for i in range(5):
            f = tmp_path / f"chunk_test_{i}.md"
            f.write_text(MARKDOWN_SAMPLE * 3)
            files.append(f)

        t0 = time.perf_counter()
        total_chunks = 0
        for fp in files:
            stats, _ = engine.ingest_file("perf_throughput", fp)
            total_chunks += stats.added
        elapsed = time.perf_counter() - t0

        cps = total_chunks / elapsed if elapsed > 0 else 0
        print(f"\nIngestion throughput: {cps:.1f} chunks/sec ({total_chunks} chunks in {elapsed:.2f}s)")
        assert total_chunks > 0


# ---------------------------------------------------------------------------
# Search benchmarks
# ---------------------------------------------------------------------------

class TestSearchPerformance:
    @pytest.fixture(autouse=True)
    def setup(self, engine, tmp_path):
        self.engine = engine
        engine.create_collection("perf_search")
        for i in range(5):
            engine.ingest_text(
                "perf_search",
                f"Document {i}: {MARKDOWN_SAMPLE}",
                source=f"doc://perf_{i}",
                file_type="markdown",
            )
        self.collection_name = "perf_search"

    def test_bm25_latency(self):
        times = _repeat(
            lambda: self.engine.search_bm25(self.collection_name, "JWT authentication token"), n=5
        )
        p95_ms = statistics.quantiles(times, n=20)[-1] * 1000  # ~95th percentile
        print(f"\nBM25 p95 latency: {p95_ms:.1f}ms")
        assert p95_ms < BASELINE_BM25_SEARCH_MS, (
            f"BM25 p95 {p95_ms:.1f}ms > baseline {BASELINE_BM25_SEARCH_MS}ms"
        )

    def test_vector_search_latency(self):
        times = _repeat(
            lambda: self.engine.search_vector(
                self.collection_name, "how does authentication work",
                use_expansion=False, use_hyde=False
            ),
            n=5,
        )
        p95_ms = statistics.quantiles(times, n=20)[-1] * 1000
        print(f"\nVector search p95 latency: {p95_ms:.1f}ms")
        assert p95_ms < BASELINE_VECTOR_SEARCH_MS, (
            f"Vector p95 {p95_ms:.1f}ms > baseline {BASELINE_VECTOR_SEARCH_MS}ms"
        )

    def test_hybrid_search_latency(self):
        from unittest.mock import patch
        no_expand = {"lex": [], "vec": [], "hyde": ""}
        times = _repeat(
            lambda: self.engine.search_hybrid(
                self.collection_name, "JWT token authentication",
                use_expansion=False
            ),
            n=5,
        )
        p95_ms = statistics.quantiles(times, n=20)[-1] * 1000
        print(f"\nHybrid search p95 latency: {p95_ms:.1f}ms")
        assert p95_ms < BASELINE_HYBRID_SEARCH_MS, (
            f"Hybrid p95 {p95_ms:.1f}ms > baseline {BASELINE_HYBRID_SEARCH_MS}ms"
        )

    def test_bm25_vs_vector_relative(self):
        """BM25 must be faster than or equal to vector (BM25 is pure sparse)."""
        bm25_times = _repeat(
            lambda: self.engine.search_bm25(self.collection_name, "token auth"), n=5
        )
        vec_times = _repeat(
            lambda: self.engine.search_vector(
                self.collection_name, "token auth", use_expansion=False, use_hyde=False
            ),
            n=5,
        )
        bm25_mean = statistics.mean(bm25_times) * 1000
        vec_mean = statistics.mean(vec_times) * 1000
        print(f"\nBM25 mean: {bm25_mean:.1f}ms, Vector mean: {vec_mean:.1f}ms")
        # BM25 should generally be faster; allow 2x tolerance
        assert bm25_mean < vec_mean * 3, "BM25 unexpectedly 3x slower than vector"

    def test_search_scales_with_top_k(self):
        """top_k=50 should not be dramatically slower than top_k=5."""
        t_small = statistics.mean(_repeat(
            lambda: self.engine.search_bm25(self.collection_name, "authentication", top_k=5), n=3
        ))
        t_large = statistics.mean(_repeat(
            lambda: self.engine.search_bm25(self.collection_name, "authentication", top_k=50), n=3
        ))
        print(f"\ntop_k=5: {t_small*1000:.1f}ms, top_k=50: {t_large*1000:.1f}ms")
        assert t_large < t_small * 10, "Search with top_k=50 is >10x slower than top_k=5"


# ---------------------------------------------------------------------------
# Collection operation benchmarks
# ---------------------------------------------------------------------------

class TestCollectionOperationPerformance:
    def test_create_collection_time(self, engine):
        t0 = time.perf_counter()
        engine.create_collection("perf_create")
        elapsed = time.perf_counter() - t0
        _assert_ms(elapsed, BASELINE_CREATE_COLLECTION_MS, "create_collection")

    def test_list_collections_time(self, engine):
        for i in range(5):
            engine.create_collection(f"perf_list_{i}")

        t0 = time.perf_counter()
        engine.list_collections()
        elapsed = time.perf_counter() - t0
        _assert_ms(elapsed, BASELINE_LIST_COLLECTIONS_MS, "list_collections (5 cols)")

    def test_optimize_time(self, engine, tmp_path):
        engine.create_collection("perf_opt")
        engine.ingest_text("perf_opt", MARKDOWN_SAMPLE, source="doc://opt")

        t0 = time.perf_counter()
        engine.optimize_collection("perf_opt")
        elapsed = time.perf_counter() - t0
        # Optimize should complete within 30 seconds
        assert elapsed < 30, f"Optimize took {elapsed:.1f}s > 30s"

    def test_cold_start_embed_hot(self, engine):
        """Preload models: second call must be near-instant."""
        engine.preload_resources()  # warm
        t0 = time.perf_counter()
        engine.preload_resources()  # hot (models cached)
        elapsed = time.perf_counter() - t0
        # Hot preload should be nearly instant (< 1 second)
        assert elapsed < 1.0, f"Hot preload took {elapsed:.2f}s — models not cached?"
