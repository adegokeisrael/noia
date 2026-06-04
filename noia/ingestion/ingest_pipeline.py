"""
noia/ingestion/ingest_pipeline.py
──────────────────────────────────
Ingestion pipeline: loads raw telecom operational documents, applies recursive
character-level chunking, computes dense embeddings, and inserts into both
ChromaDB (dense vector store) and a BM25 inverted index (sparse retrieval).

Supports incremental re-ingestion: documents already present in the index
(matched by doc_id) are skipped unless --force-reindex is passed.

Usage (CLI)::

    python -m noia.ingestion.ingest_pipeline --corpus-dir data/corpus

Usage (API)::

    from noia.ingestion.ingest_pipeline import IngestPipeline
    pipeline = IngestPipeline()
    pipeline.ingest_directory("data/corpus")
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import typer
from rich.console import Console
from rich.progress import track

logger = logging.getLogger(__name__)
console = Console()
app = typer.Typer(help="Ingest telecom corpus into NOIA vector + BM25 index.")

# ── Lazy imports (heavy ML deps) ─────────────────────────────────────────────

def _get_chroma_client(persist_dir: str):
    import chromadb  # noqa: PLC0415
    return chromadb.PersistentClient(path=persist_dir)


def _get_embedding_fn(model_name: str):
    from chromadb.utils.embedding_functions import (  # noqa: PLC0415
        SentenceTransformerEmbeddingFunction,
    )
    return SentenceTransformerEmbeddingFunction(model_name=model_name)


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class DocumentChunk:
    """A single text chunk derived from a source document."""

    chunk_id: str           # SHA-256(doc_id + chunk_index)
    doc_id: str
    source_type: str        # runbook | incident | sla | maintenance
    source_file: str
    title: str
    timestamp: str
    chunk_index: int
    total_chunks: int
    text: str
    tokens: list[str] = field(default_factory=list)  # for BM25

    def to_chroma_metadata(self) -> dict:
        return {
            "doc_id":       self.doc_id,
            "source_type":  self.source_type,
            "source_file":  self.source_file,
            "title":        self.title,
            "timestamp":    self.timestamp,
            "chunk_index":  self.chunk_index,
            "total_chunks": self.total_chunks,
        }


@dataclass
class IngestReport:
    """Summary statistics produced by the ingestion pipeline."""

    started_at: str = ""
    completed_at: str = ""
    documents_scanned: int = 0
    documents_skipped: int = 0
    documents_ingested: int = 0
    chunks_created: int = 0
    index_size_dense: int = 0
    index_size_sparse: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "started_at":        self.started_at,
            "completed_at":      self.completed_at,
            "documents_scanned": self.documents_scanned,
            "documents_skipped": self.documents_skipped,
            "documents_ingested":self.documents_ingested,
            "chunks_created":    self.chunks_created,
            "index_size_dense":  self.index_size_dense,
            "index_size_sparse": self.index_size_sparse,
            "errors":            self.errors,
        }


# ── Text splitter ─────────────────────────────────────────────────────────────

class RecursiveCharacterSplitter:
    """
    Splits text into overlapping chunks using a hierarchy of separators,
    similar to LangChain's RecursiveCharacterTextSplitter but dependency-free.

    Parameters
    ----------
    chunk_size : int
        Maximum number of characters per chunk.
    chunk_overlap : int
        Number of characters to overlap between consecutive chunks.
    separators : list[str]
        Separator hierarchy (tried in order).
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def split(self, text: str) -> list[str]:
        """Return a list of text chunks."""
        return list(self._split_recursive(text, self.separators))

    def _split_recursive(self, text: str, separators: list[str]) -> Iterator[str]:
        if len(text) <= self.chunk_size:
            if text.strip():
                yield text.strip()
            return

        separator = separators[0] if len(separators) > 1 else separators[0]
        for sep in separators:
            if sep in text:
                separator = sep
                break

        parts = text.split(separator)
        current = ""

        for part in parts:
            candidate = current + (separator if current else "") + part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current.strip():
                    yield current.strip()
                # Handle overlap
                overlap_text = current[-self.chunk_overlap:] if current else ""
                current = overlap_text + (separator if overlap_text else "") + part
                if len(current) > self.chunk_size and len(separators) > 1:
                    yield from self._split_recursive(current, separators[1:])
                    current = ""

        if current.strip():
            yield current.strip()


# ── BM25 index wrapper ────────────────────────────────────────────────────────

class BM25Index:
    """
    Wrapper around rank_bm25.BM25Okapi that supports persistence and
    incremental updates.

    The index stores:
      - ``corpus``     : list of tokenised chunk texts
      - ``chunk_ids``  : parallel list of chunk_id strings
      - ``metadata``   : parallel list of metadata dicts
    """

    def __init__(self) -> None:
        self.corpus: list[list[str]] = []
        self.chunk_ids: list[str] = []
        self.metadata: list[dict] = []
        self._bm25 = None

    def _tokenise(self, text: str) -> list[str]:
        """Lower-case, punctuation-stripped tokenisation."""
        import re  # noqa: PLC0415
        tokens = re.sub(r"[^a-zA-Z0-9\-_./]", " ", text.lower()).split()
        return [t for t in tokens if len(t) > 1]

    def add(self, chunk_id: str, text: str, metadata: dict) -> None:
        """Add a single chunk to the index."""
        if chunk_id in self.chunk_ids:
            return
        self.chunk_ids.append(chunk_id)
        self.corpus.append(self._tokenise(text))
        self.metadata.append(metadata)
        self._bm25 = None  # invalidate cached BM25 object

    def _ensure_built(self) -> None:
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi  # noqa: PLC0415
            self._bm25 = BM25Okapi(self.corpus)

    def query(self, query_text: str, top_k: int = 20) -> list[dict]:
        """
        Retrieve the top-k most relevant chunks for ``query_text``.

        Returns
        -------
        list[dict]
            Each dict contains 'chunk_id', 'score', 'text_tokens', 'metadata'.
        """
        if not self.corpus:
            return []
        self._ensure_built()
        tokens = self._tokenise(query_text)
        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {
                "chunk_id": self.chunk_ids[idx],
                "score":    float(scores[idx]),
                "metadata": self.metadata[idx],
            }
            for idx in top_indices
            if scores[idx] > 0.0
        ]

    def save(self, path: str | Path) -> None:
        """Persist index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump({
                "corpus":     self.corpus,
                "chunk_ids":  self.chunk_ids,
                "metadata":   self.metadata,
            }, fh)
        logger.info("BM25 index saved to %s (%d chunks).", path, len(self.chunk_ids))

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        """Load a persisted index from disk."""
        path = Path(path)
        if not path.exists():
            logger.warning("BM25 index not found at %s — returning empty index.", path)
            return cls()
        with path.open("rb") as fh:
            data = pickle.load(fh)
        idx = cls()
        idx.corpus    = data["corpus"]
        idx.chunk_ids = data["chunk_ids"]
        idx.metadata  = data["metadata"]
        logger.info("BM25 index loaded from %s (%d chunks).", path, len(idx.chunk_ids))
        return idx


# ── Main pipeline ─────────────────────────────────────────────────────────────

class IngestPipeline:
    """
    Orchestrates end-to-end ingestion of telecom operational documents.

    Parameters
    ----------
    chroma_persist_dir : str
        ChromaDB persistence directory.
    chroma_collection : str
        Name of the ChromaDB collection.
    bm25_index_path : str
        Path to persist the BM25 index.
    embedding_model : str
        HuggingFace sentence-transformers model name.
    chunk_size : int
        Splitter chunk size in characters.
    chunk_overlap : int
        Splitter overlap in characters.
    batch_size : int
        Number of chunks per ChromaDB insertion batch.
    """

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".html"}

    def __init__(
        self,
        chroma_persist_dir: str = "data/chroma",
        chroma_collection: str = "noia_knowledge_base",
        bm25_index_path: str = "data/bm25_index.pkl",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        batch_size: int = 64,
    ) -> None:
        self.chroma_persist_dir  = chroma_persist_dir
        self.chroma_collection   = chroma_collection
        self.bm25_index_path     = Path(bm25_index_path)
        self.embedding_model     = embedding_model
        self.batch_size          = batch_size

        self.splitter = RecursiveCharacterSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Lazy-initialised heavy components
        self._chroma_client     = None
        self._chroma_collection = None
        self._bm25: BM25Index | None = None

        logger.info("IngestPipeline initialised (embedding=%s, chunk=%d/%d).",
                    embedding_model, chunk_size, chunk_overlap)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_collection(self):
        if self._chroma_collection is None:
            client = _get_chroma_client(self.chroma_persist_dir)
            emb_fn = _get_embedding_fn(self.embedding_model)
            self._chroma_collection = client.get_or_create_collection(
                name=self.chroma_collection,
                embedding_function=emb_fn,
                metadata={"hnsw:space": "cosine"},
            )
        return self._chroma_collection

    def _get_bm25(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = BM25Index.load(self.bm25_index_path)
        return self._bm25

    @staticmethod
    def _chunk_id(doc_id: str, chunk_index: int) -> str:
        return hashlib.sha256(f"{doc_id}::{chunk_index}".encode()).hexdigest()[:16]

    def _load_text(self, path: Path) -> str:
        """Load plain text from a file, supporting .txt, .md, .pdf, .docx, .html."""
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            return path.read_text(encoding="utf-8", errors="replace")
        if suffix == ".pdf":
            try:
                from pypdf import PdfReader  # noqa: PLC0415
                reader = PdfReader(str(path))
                return "\n\n".join(p.extract_text() or "" for p in reader.pages)
            except Exception as exc:  # noqa: BLE001
                logger.warning("PDF read failed for %s: %s", path, exc)
                return ""
        if suffix == ".docx":
            try:
                from docx import Document  # noqa: PLC0415
                doc = Document(str(path))
                return "\n\n".join(p.text for p in doc.paragraphs)
            except Exception as exc:  # noqa: BLE001
                logger.warning("DOCX read failed for %s: %s", path, exc)
                return ""
        if suffix == ".html":
            try:
                from bs4 import BeautifulSoup  # noqa: PLC0415
                soup = BeautifulSoup(path.read_bytes(), "lxml")
                return soup.get_text(separator="\n")
            except Exception as exc:  # noqa: BLE001
                logger.warning("HTML read failed for %s: %s", path, exc)
                return ""
        return ""

    def _parse_header(self, text: str, source_file: str) -> dict:
        """
        Extract structured metadata from NOIA-format document headers.
        Falls back to sensible defaults for non-standard documents.
        """
        meta = {
            "doc_id":      Path(source_file).stem,
            "source_type": "unknown",
            "title":       Path(source_file).stem,
            "timestamp":   datetime.now(tz=timezone.utc).isoformat(),
        }
        for line in text.splitlines()[:8]:
            if line.startswith("DOC_ID:"):
                meta["doc_id"] = line.split(":", 1)[1].strip()
            elif line.startswith("SOURCE_TYPE:"):
                meta["source_type"] = line.split(":", 1)[1].strip()
            elif line.startswith("TITLE:"):
                meta["title"] = line.split(":", 1)[1].strip()
            elif line.startswith("TIMESTAMP:"):
                meta["timestamp"] = line.split(":", 1)[1].strip()
        return meta

    def _already_indexed(self, doc_id: str) -> bool:
        """Check whether any chunk with this doc_id exists in ChromaDB."""
        col = self._get_collection()
        results = col.get(where={"doc_id": doc_id}, limit=1, include=[])
        return len(results["ids"]) > 0

    def _process_file(self, path: Path) -> list[DocumentChunk]:
        """Load, parse, and chunk a single file."""
        text = self._load_text(path)
        if not text.strip():
            return []
        meta = self._parse_header(text, str(path))
        raw_chunks = self.splitter.split(text)
        chunks = []
        for idx, chunk_text in enumerate(raw_chunks):
            cid = self._chunk_id(meta["doc_id"], idx)
            chunks.append(DocumentChunk(
                chunk_id    = cid,
                doc_id      = meta["doc_id"],
                source_type = meta["source_type"],
                source_file = str(path),
                title       = meta["title"],
                timestamp   = meta["timestamp"],
                chunk_index = idx,
                total_chunks= len(raw_chunks),
                text        = chunk_text,
            ))
        return chunks

    def _insert_batch(self, chunks: list[DocumentChunk]) -> None:
        """Insert a batch of chunks into ChromaDB and BM25 index."""
        col   = self._get_collection()
        bm25  = self._get_bm25()
        ids   = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        metas = [c.to_chroma_metadata() for c in chunks]
        col.add(ids=ids, documents=texts, metadatas=metas)
        for chunk in chunks:
            bm25.add(chunk.chunk_id, chunk.text, chunk.to_chroma_metadata())

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest_file(self, path: str | Path, force: bool = False) -> int:
        """
        Ingest a single document file.

        Parameters
        ----------
        path : str | Path
            Path to the document.
        force : bool
            Re-ingest even if the document is already indexed.

        Returns
        -------
        int
            Number of chunks inserted.
        """
        path = Path(path)
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            logger.debug("Skipping unsupported file type: %s", path)
            return 0

        chunks = self._process_file(path)
        if not chunks:
            return 0

        doc_id = chunks[0].doc_id
        if not force and self._already_indexed(doc_id):
            logger.debug("Skipping already-indexed document: %s", doc_id)
            return 0

        for i in range(0, len(chunks), self.batch_size):
            self._insert_batch(chunks[i : i + self.batch_size])

        logger.info("Ingested %s → %d chunks.", path.name, len(chunks))
        return len(chunks)

    def ingest_directory(
        self,
        directory: str | Path,
        force: bool = False,
        recursive: bool = True,
    ) -> IngestReport:
        """
        Ingest all supported documents in a directory.

        Parameters
        ----------
        directory : str | Path
            Root directory to scan.
        force : bool
            Re-ingest documents already in the index.
        recursive : bool
            Whether to recurse into subdirectories.

        Returns
        -------
        IngestReport
            Ingestion statistics.
        """
        directory = Path(directory)
        report = IngestReport(
            started_at=datetime.now(tz=timezone.utc).isoformat()
        )

        glob_fn = directory.rglob if recursive else directory.glob
        all_files = [
            p for p in glob_fn("*")
            if p.is_file()
            and p.suffix.lower() in self.SUPPORTED_EXTENSIONS
            and not p.name.endswith(".meta.json")
        ]

        report.documents_scanned = len(all_files)
        console.print(f"[bold]Found {len(all_files)} files in {directory}[/bold]")

        for path in track(all_files, description="[cyan]Ingesting documents...[/cyan]"):
            try:
                inserted = self.ingest_file(path, force=force)
                if inserted == 0:
                    report.documents_skipped += 1
                else:
                    report.documents_ingested += 1
                    report.chunks_created += inserted
            except Exception as exc:  # noqa: BLE001
                msg = f"{path}: {exc}"
                logger.error("Ingestion error — %s", msg)
                report.errors.append(msg)

        # Persist BM25 index
        bm25 = self._get_bm25()
        bm25.save(self.bm25_index_path)

        col = self._get_collection()
        report.index_size_dense  = col.count()
        report.index_size_sparse = len(bm25.chunk_ids)
        report.completed_at = datetime.now(tz=timezone.utc).isoformat()

        console.print(
            f"[bold green]✓ Ingestion complete[/bold green]\n"
            f"  Ingested: {report.documents_ingested} | "
            f"Skipped: {report.documents_skipped} | "
            f"Chunks: {report.chunks_created} | "
            f"Errors: {len(report.errors)}"
        )
        return report

    def get_index_stats(self) -> dict:
        """Return current index size statistics."""
        col  = self._get_collection()
        bm25 = self._get_bm25()
        return {
            "dense_chunks":  col.count(),
            "sparse_chunks": len(bm25.chunk_ids),
            "collection":    self.chroma_collection,
            "bm25_path":     str(self.bm25_index_path),
        }


# ── CLI entrypoint ────────────────────────────────────────────────────────────

@app.command()
def main(
    corpus_dir: Path = typer.Option(Path("data/corpus"), help="Corpus directory."),
    force: bool = typer.Option(False, help="Re-ingest already-indexed documents."),
    report_out: Path = typer.Option(None, help="Write ingestion report JSON here."),
) -> None:
    """Ingest telecom corpus into ChromaDB + BM25 index."""
    from config.settings import get_settings  # noqa: PLC0415
    s = get_settings()
    pipeline = IngestPipeline(
        chroma_persist_dir = str(s.chroma_persist_dir),
        chroma_collection  = s.chroma_collection_name,
        bm25_index_path    = str(s.bm25_index_path),
        embedding_model    = s.embedding_model,
        chunk_size         = s.chunk_size,
        chunk_overlap      = s.chunk_overlap,
    )
    report = pipeline.ingest_directory(corpus_dir, force=force)
    if report_out:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report.to_dict(), indent=2))
        console.print(f"[dim]Report written to {report_out}[/dim]")


if __name__ == "__main__":
    app()
