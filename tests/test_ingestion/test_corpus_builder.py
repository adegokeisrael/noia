"""
tests/test_ingestion/test_corpus_builder.py
────────────────────────────────────────────
Unit tests for the synthetic corpus builder.
"""

import json
import tempfile
from pathlib import Path

import pytest

from noia.ingestion.corpus_builder import CorpusBuilder


@pytest.fixture
def tmp_corpus(tmp_path):
    builder = CorpusBuilder(output_dir=tmp_path / "corpus", seed=0)
    return builder, tmp_path / "corpus"


class TestCorpusBuilder:
    def test_build_returns_correct_count(self, tmp_corpus):
        builder, output_dir = tmp_corpus
        files = builder.build(total=20)
        assert len(files) == 20

    def test_all_files_exist_on_disk(self, tmp_corpus):
        builder, output_dir = tmp_corpus
        files = builder.build(total=10)
        for f in files:
            assert f.exists(), f"File missing: {f}"

    def test_subdirectories_created(self, tmp_corpus):
        builder, output_dir = tmp_corpus
        builder.build(total=20)
        expected_dirs = {"runbook", "incident", "sla", "maintenance"}
        actual_dirs   = {p.name for p in output_dir.iterdir() if p.is_dir()}
        assert expected_dirs == actual_dirs

    def test_sidecar_metadata_files_exist(self, tmp_corpus):
        builder, output_dir = tmp_corpus
        files = builder.build(total=5)
        for f in files:
            meta_path = f.with_suffix("").with_suffix(".meta.json")
            # meta file pattern: RB-0001.meta.json alongside RB-0001.txt
            meta_path = f.parent / (f.stem + ".meta.json")
            assert meta_path.exists(), f"Metadata missing: {meta_path}"

    def test_metadata_json_structure(self, tmp_corpus):
        builder, output_dir = tmp_corpus
        files = builder.build(total=5)
        for f in files:
            meta_path = f.parent / (f.stem + ".meta.json")
            with meta_path.open() as fh:
                meta = json.load(fh)
            assert "doc_id" in meta
            assert "source_type" in meta
            assert "timestamp" in meta
            assert meta["source_type"] in {"runbook", "incident", "sla", "maintenance"}

    def test_reproducibility_with_same_seed(self, tmp_path):
        b1 = CorpusBuilder(output_dir=tmp_path / "c1", seed=42)
        b2 = CorpusBuilder(output_dir=tmp_path / "c2", seed=42)
        f1 = b1.build(total=5)
        f2 = b2.build(total=5)
        # File contents should match
        for p1, p2 in zip(sorted(f1), sorted(f2)):
            assert p1.read_text() == p2.read_text()

    def test_different_seeds_produce_different_content(self, tmp_path):
        b1 = CorpusBuilder(output_dir=tmp_path / "c1", seed=1)
        b2 = CorpusBuilder(output_dir=tmp_path / "c2", seed=2)
        f1 = b1.build(total=5)
        f2 = b2.build(total=5)
        texts = [p.read_text() for p in f1], [p.read_text() for p in f2]
        # At least one file should differ
        assert not all(a == b for a, b in zip(texts[0], texts[1]))

    def test_document_headers_present(self, tmp_corpus):
        builder, output_dir = tmp_corpus
        files = builder.build(total=5)
        for f in files:
            content = f.read_text()
            assert "DOC_ID:" in content
            assert "SOURCE_TYPE:" in content
            assert "TITLE:" in content
            assert "TIMESTAMP:" in content
