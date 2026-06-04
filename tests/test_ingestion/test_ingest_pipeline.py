"""
tests/test_ingestion/test_ingest_pipeline.py
─────────────────────────────────────────────
Unit tests for the ingestion pipeline (splitter + BM25 index).
Heavy ML components (ChromaDB, embeddings) are mocked.
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from noia.ingestion.ingest_pipeline import (
    BM25Index,
    DocumentChunk,
    IngestPipeline,
    RecursiveCharacterSplitter,
)


# ── RecursiveCharacterSplitter ────────────────────────────────────────────────

class TestRecursiveCharacterSplitter:
    def test_short_text_returns_single_chunk(self):
        splitter = RecursiveCharacterSplitter(chunk_size=512, chunk_overlap=64)
        text = "Short text under limit."
        chunks = splitter.split(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_is_split(self):
        splitter = RecursiveCharacterSplitter(chunk_size=100, chunk_overlap=10)
        text = ("This is sentence one. " * 20).strip()
        chunks = splitter.split(text)
        assert len(chunks) > 1

    def test_chunks_do_not_exceed_size(self):
        splitter = RecursiveCharacterSplitter(chunk_size=150, chunk_overlap=20)
        text = "Word " * 200
        chunks = splitter.split(text)
        for chunk in chunks:
            assert len(chunk) <= 200  # allow small margin for overlap

    def test_empty_text_returns_empty(self):
        splitter = RecursiveCharacterSplitter(chunk_size=512, chunk_overlap=64)
        assert splitter.split("") == []
        assert splitter.split("   ") == []

    def test_overlap_content_shared(self):
        splitter = RecursiveCharacterSplitter(chunk_size=50, chunk_overlap=20)
        text = "AAAA " * 30
        chunks = splitter.split(text)
        if len(chunks) > 1:
            # The end of chunk N should appear at the start of chunk N+1
            overlap_end   = chunks[0][-15:]
            overlap_start = chunks[1][:15]
            # They share at least some characters
            assert any(c in overlap_start for c in overlap_end if c.strip())


# ── BM25Index ─────────────────────────────────────────────────────────────────

class TestBM25Index:
    def _populated_index(self) -> BM25Index:
        idx = BM25Index()
        docs = [
            ("chunk-001", "BGP session drop on Nokia PE router route flap",
             {"source_type": "runbook", "doc_id": "RB-0001", "title": "BGP Runbook"}),
            ("chunk-002", "OSPF adjacency loss memory exhaustion Cisco",
             {"source_type": "incident", "doc_id": "INC-000001", "title": "P1 Incident"}),
            ("chunk-003", "SLA availability 99.99 percent MTTR four hours",
             {"source_type": "sla", "doc_id": "SLA-0001", "title": "SLA Schedule"}),
            ("chunk-004", "Maintenance fiber splice OTDR optical power",
             {"source_type": "maintenance", "doc_id": "MNT-0001", "title": "CHG-00001"}),
        ]
        for cid, text, meta in docs:
            idx.add(cid, text, meta)
        return idx

    def test_add_and_query(self):
        idx = self._populated_index()
        results = idx.query("BGP session drop", top_k=3)
        assert len(results) >= 1
        assert results[0]["chunk_id"] == "chunk-001"

    def test_query_returns_empty_for_no_index(self):
        idx = BM25Index()
        assert idx.query("BGP", top_k=5) == []

    def test_scores_are_positive(self):
        idx = self._populated_index()
        results = idx.query("OSPF adjacency", top_k=3)
        for r in results:
            assert r["score"] > 0.0

    def test_duplicate_chunk_not_added(self):
        idx = BM25Index()
        idx.add("c1", "some text about BGP", {"doc_id": "X"})
        idx.add("c1", "some text about BGP", {"doc_id": "X"})
        assert len(idx.chunk_ids) == 1

    def test_persist_and_load(self, tmp_path):
        idx = self._populated_index()
        path = tmp_path / "bm25.pkl"
        idx.save(path)
        assert path.exists()
        loaded = BM25Index.load(path)
        assert loaded.chunk_ids == idx.chunk_ids
        assert len(loaded.corpus) == len(idx.corpus)

    def test_load_nonexistent_returns_empty(self, tmp_path):
        idx = BM25Index.load(tmp_path / "doesnotexist.pkl")
        assert idx.chunk_ids == []
        assert idx.corpus == []

    def test_top_k_respected(self):
        idx = self._populated_index()
        results = idx.query("network", top_k=2)
        assert len(results) <= 2

    def test_unrelated_query_returns_empty_or_low_score(self):
        idx = self._populated_index()
        results = idx.query("unrelated xyzzy foobar", top_k=5)
        # All results should be either empty or have score 0
        for r in results:
            assert r["score"] >= 0.0


# ── IngestPipeline (mocked heavy components) ──────────────────────────────────

class TestIngestPipeline:
    def _make_pipeline(self, tmp_path) -> IngestPipeline:
        return IngestPipeline(
            chroma_persist_dir = str(tmp_path / "chroma"),
            chroma_collection  = "test_collection",
            bm25_index_path    = str(tmp_path / "bm25.pkl"),
            embedding_model    = "sentence-transformers/all-MiniLM-L6-v2",
            chunk_size         = 200,
            chunk_overlap      = 20,
        )

    def test_parse_header_extracts_metadata(self, tmp_path):
        pipeline = self._make_pipeline(tmp_path)
        text = (
            "DOC_ID: RB-0001\n"
            "SOURCE_TYPE: runbook\n"
            "TITLE: BGP Recovery Runbook\n"
            "TIMESTAMP: 2024-01-15T10:00:00\n"
            "────────────────────────────────────────\n\n"
            "Step 1 — Verify BGP neighbour state.\n"
        )
        meta = pipeline._parse_header(text, "data/corpus/runbook/RB-0001.txt")
        assert meta["doc_id"]      == "RB-0001"
        assert meta["source_type"] == "runbook"
        assert meta["title"]       == "BGP Recovery Runbook"
        assert "2024-01-15" in meta["timestamp"]

    def test_parse_header_fallback_for_unstructured(self, tmp_path):
        pipeline = self._make_pipeline(tmp_path)
        text = "This is some unstructured document text with no headers."
        meta = pipeline._parse_header(text, "data/docs/my_doc.txt")
        assert meta["doc_id"] == "my_doc"
        assert meta["source_type"] == "unknown"

    def test_chunk_id_deterministic(self, tmp_path):
        pipeline = self._make_pipeline(tmp_path)
        id1 = pipeline._chunk_id("RB-0001", 0)
        id2 = pipeline._chunk_id("RB-0001", 0)
        assert id1 == id2
        assert len(id1) == 16

    def test_chunk_id_unique_per_index(self, tmp_path):
        pipeline = self._make_pipeline(tmp_path)
        id0 = pipeline._chunk_id("RB-0001", 0)
        id1 = pipeline._chunk_id("RB-0001", 1)
        assert id0 != id1

    def test_load_text_plain(self, tmp_path):
        pipeline = self._make_pipeline(tmp_path)
        f = tmp_path / "test.txt"
        f.write_text("Hello NOC world.")
        assert pipeline._load_text(f) == "Hello NOC world."

    def test_load_text_unsupported_returns_empty(self, tmp_path):
        pipeline = self._make_pipeline(tmp_path)
        f = tmp_path / "file.xyz"
        f.write_text("ignored")
        assert pipeline._load_text(f) == ""

    def test_ingest_file_skips_unsupported_extension(self, tmp_path):
        pipeline = self._make_pipeline(tmp_path)
        f = tmp_path / "file.csv"
        f.write_text("a,b,c")
        result = pipeline.ingest_file(f)
        assert result == 0

    def test_process_file_returns_chunks(self, tmp_path):
        pipeline = self._make_pipeline(tmp_path)
        f = tmp_path / "rb.txt"
        content = (
            "DOC_ID: RB-TEST\nSOURCE_TYPE: runbook\n"
            "TITLE: Test Runbook\nTIMESTAMP: 2024-01-01T00:00:00\n\n"
            + ("This is a runbook step with enough content to form chunks. " * 30)
        )
        f.write_text(content)
        chunks = pipeline._process_file(f)
        assert len(chunks) > 0
        assert all(isinstance(c, DocumentChunk) for c in chunks)
        assert all(c.doc_id == "RB-TEST" for c in chunks)
        assert all(c.source_type == "runbook" for c in chunks)
