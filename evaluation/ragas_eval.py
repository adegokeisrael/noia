"""
evaluation/ragas_eval.py
─────────────────────────
RAGAS evaluation pipeline for NOIA.

Evaluates the full pipeline (hybrid retrieval + reranker + LLM) on a
held-out benchmark of 100 query–reference pairs using four RAGAS metrics:

  - Faithfulness      : Are response claims supported by retrieved context?
  - Answer Relevancy  : Is the answer relevant to the question?
  - Context Precision : Are retrieved chunks relevant to the question?
  - Context Recall    : Do retrieved chunks contain the reference answer?

Usage::

    python -m evaluation.ragas_eval \\
        --benchmark evaluation/benchmark_queries.json \\
        --output results/ragas_report.json

The benchmark JSON must be a list of objects with fields:
  - "query"       : engineer query string
  - "reference"   : reference (ground-truth) answer string
  - "mode"        : task mode ("query" | "rca" | "summarise" | etc.)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from config.settings import get_settings
from noia.reasoning.reasoning_engine import ReasoningEngine
from noia.retrieval.hybrid_retriever import HybridRetriever
from noia.retrieval.reranker import Reranker

logger  = logging.getLogger(__name__)
console = Console()
app     = typer.Typer(help="Run RAGAS evaluation on the NOIA benchmark.")
settings = get_settings()


# ── Metric implementations (lightweight, dependency-free) ─────────────────────

def _embed_texts(texts: list[str], model_name: str) -> "np.ndarray":
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    model = SentenceTransformer(model_name)
    import numpy as np  # noqa: PLC0415
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def cosine_sim(a, b) -> float:
    import numpy as np  # noqa: PLC0415
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_faithfulness(
    response_text: str,
    context_texts: list[str],
    embedding_model: str,
) -> float:
    """
    Faithfulness: fraction of response sentences with max context similarity ≥ 0.70.
    """
    import re  # noqa: PLC0415
    sentences = [s.strip() for s in re.split(r"[.!?]", response_text) if len(s.strip()) > 20]
    if not sentences or not context_texts:
        return 1.0

    all_texts = sentences + context_texts
    embeddings = _embed_texts(all_texts, embedding_model)
    s_embs = embeddings[:len(sentences)]
    c_embs = embeddings[len(sentences):]

    grounded = sum(
        1 for se in s_embs
        if any(cosine_sim(se, ce) >= 0.70 for ce in c_embs)
    )
    return grounded / len(sentences)


def compute_answer_relevancy(
    query: str,
    response_text: str,
    embedding_model: str,
) -> float:
    """
    Answer Relevancy: cosine similarity between query embedding and response embedding.
    """
    embs = _embed_texts([query, response_text], embedding_model)
    return max(0.0, cosine_sim(embs[0], embs[1]))


def compute_context_precision(
    query: str,
    context_texts: list[str],
    reference: str,
    embedding_model: str,
) -> float:
    """
    Context Precision: fraction of retrieved chunks whose similarity to the
    reference answer exceeds 0.65.
    """
    if not context_texts:
        return 0.0
    all_texts = [reference] + context_texts
    embs      = _embed_texts(all_texts, embedding_model)
    ref_emb   = embs[0]
    ctx_embs  = embs[1:]
    relevant  = sum(1 for ce in ctx_embs if cosine_sim(ref_emb, ce) >= 0.65)
    return relevant / len(ctx_embs)


def compute_context_recall(
    reference: str,
    context_texts: list[str],
    embedding_model: str,
) -> float:
    """
    Context Recall: max cosine similarity between the reference answer and
    any retrieved chunk (measures whether the needed information is present).
    """
    if not context_texts:
        return 0.0
    all_texts = [reference] + context_texts
    embs      = _embed_texts(all_texts, embedding_model)
    ref_emb   = embs[0]
    ctx_embs  = embs[1:]
    return max(cosine_sim(ref_emb, ce) for ce in ctx_embs)


# ── Per-sample result ─────────────────────────────────────────────────────────

@dataclass
class EvalSample:
    query:              str
    mode:               str
    reference:          str
    response_text:      str
    faithfulness:       float
    answer_relevancy:   float
    context_precision:  float
    context_recall:     float
    response_time_ms:   float
    groundedness_score: float
    review_required:    bool
    error:              str = ""


@dataclass
class EvalReport:
    benchmark_path:          str
    n_samples:               int
    n_errors:                int
    mean_faithfulness:       float
    mean_answer_relevancy:   float
    mean_context_precision:  float
    mean_context_recall:     float
    std_faithfulness:        float
    std_answer_relevancy:    float
    std_context_precision:   float
    std_context_recall:      float
    mean_groundedness:       float
    mean_latency_ms:         float
    samples:                 list[dict]

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ── Evaluation runner ─────────────────────────────────────────────────────────

class RAGASEvaluator:
    """
    Runs the NOIA pipeline on each benchmark sample and computes RAGAS metrics.

    Parameters
    ----------
    retriever : HybridRetriever
    reranker  : Reranker
    engine    : ReasoningEngine
    embedding_model : str
        Sentence-transformers model for metric computation.
    top_k : int
        Chunks to retrieve per sample.
    """

    def __init__(
        self,
        retriever:       HybridRetriever,
        reranker:        Reranker,
        engine:          ReasoningEngine,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k:           int = 5,
    ) -> None:
        self.retriever       = retriever
        self.reranker        = reranker
        self.engine          = engine
        self.embedding_model = embedding_model
        self.top_k           = top_k

    def evaluate_sample(self, sample: dict) -> EvalSample:
        query     = sample["query"]
        reference = sample["reference"]
        mode      = sample.get("mode", "query")
        t0        = time.perf_counter()
        error_msg = ""

        try:
            candidates = self.retriever.retrieve(query, top_k=settings.retrieval_top_k)
            chunks     = self.reranker.rerank(query, candidates, top_n=self.top_k)
            noia_r     = self.engine.reason(query, chunks, mode=mode)

            response_text  = noia_r.raw_text
            context_texts  = [c.get("text", "") for c in noia_r.context_chunks]
            groundedness   = noia_r.groundedness.score
            review_flag    = noia_r.groundedness.review_required

            faithfulness      = compute_faithfulness(response_text, context_texts, self.embedding_model)
            answer_relevancy  = compute_answer_relevancy(query, response_text, self.embedding_model)
            ctx_precision     = compute_context_precision(query, context_texts, reference, self.embedding_model)
            ctx_recall        = compute_context_recall(reference, context_texts, self.embedding_model)

        except Exception as exc:  # noqa: BLE001
            logger.error("Sample evaluation failed: %s", exc)
            response_text = ""
            faithfulness = answer_relevancy = ctx_precision = ctx_recall = 0.0
            groundedness = 0.0
            review_flag  = True
            error_msg    = str(exc)

        elapsed = (time.perf_counter() - t0) * 1000

        return EvalSample(
            query             = query,
            mode              = mode,
            reference         = reference,
            response_text     = response_text[:500],
            faithfulness      = faithfulness,
            answer_relevancy  = answer_relevancy,
            context_precision = ctx_precision,
            context_recall    = ctx_recall,
            response_time_ms  = elapsed,
            groundedness_score= groundedness,
            review_required   = review_flag,
            error             = error_msg,
        )

    def evaluate(self, benchmark: list[dict]) -> EvalReport:
        """Run full evaluation and return an EvalReport."""
        from rich.progress import track  # noqa: PLC0415
        samples = [
            self.evaluate_sample(s)
            for s in track(benchmark, description="[cyan]Evaluating samples…[/cyan]")
        ]

        valid = [s for s in samples if not s.error]
        n_err = len(samples) - len(valid)

        def _mean(vals): return mean(vals) if vals else 0.0
        def _std(vals):  return stdev(vals) if len(vals) > 1 else 0.0

        fa  = [s.faithfulness      for s in valid]
        ar  = [s.answer_relevancy  for s in valid]
        cp  = [s.context_precision for s in valid]
        cr  = [s.context_recall    for s in valid]
        gr  = [s.groundedness_score for s in valid]
        lt  = [s.response_time_ms  for s in samples]

        return EvalReport(
            benchmark_path        = "",
            n_samples             = len(samples),
            n_errors              = n_err,
            mean_faithfulness     = _mean(fa),
            mean_answer_relevancy = _mean(ar),
            mean_context_precision= _mean(cp),
            mean_context_recall   = _mean(cr),
            std_faithfulness      = _std(fa),
            std_answer_relevancy  = _std(ar),
            std_context_precision = _std(cp),
            std_context_recall    = _std(cr),
            mean_groundedness     = _mean(gr),
            mean_latency_ms       = _mean(lt),
            samples               = [asdict(s) for s in samples],
        )


def _print_report(report: EvalReport) -> None:
    table = Table(title="NOIA RAGAS Evaluation Results", show_lines=True)
    table.add_column("Metric",          style="bold cyan",  width=28)
    table.add_column("Mean",            style="bold green", width=10)
    table.add_column("Std Dev",         style="yellow",     width=10)

    rows = [
        ("Faithfulness",       report.mean_faithfulness,     report.std_faithfulness),
        ("Answer Relevancy",   report.mean_answer_relevancy, report.std_answer_relevancy),
        ("Context Precision",  report.mean_context_precision,report.std_context_precision),
        ("Context Recall",     report.mean_context_recall,   report.std_context_recall),
        ("Groundedness G(r)",  report.mean_groundedness,     0.0),
        ("Latency (ms)",       report.mean_latency_ms,       0.0),
    ]
    for name, m, s in rows:
        table.add_row(name, f"{m:.4f}", f"{s:.4f}" if s else "—")

    console.print()
    console.print(table)
    console.print(
        f"\n[bold]Samples:[/bold] {report.n_samples} | "
        f"[bold]Errors:[/bold] {report.n_errors}"
    )


@app.command()
def main(
    benchmark: Path = typer.Option(
        Path("evaluation/benchmark_queries.json"),
        help="Path to benchmark JSON file.",
    ),
    output: Path = typer.Option(
        Path("results/ragas_report.json"),
        help="Path to write evaluation report JSON.",
    ),
    top_k: int = typer.Option(5, help="Chunks per query."),
) -> None:
    """Run RAGAS evaluation on the NOIA pipeline."""
    if not benchmark.exists():
        console.print(f"[red]Benchmark file not found: {benchmark}[/red]")
        raise typer.Exit(1)

    with benchmark.open() as fh:
        benchmark_data = json.load(fh)

    console.print(f"[bold]Loaded {len(benchmark_data)} benchmark samples.[/bold]")

    retriever = HybridRetriever(
        chroma_persist_dir = str(settings.chroma_persist_dir),
        chroma_collection  = settings.chroma_collection_name,
        bm25_index_path    = str(settings.bm25_index_path),
        embedding_model    = settings.embedding_model,
        hybrid_alpha       = settings.hybrid_alpha,
    )
    reranker = Reranker(model_name=settings.reranker_model, top_n=top_k)
    engine   = ReasoningEngine(
        llm_backend      = settings.llm_backend,
        local_model_path = str(settings.local_model_path),
        openai_api_base  = settings.openai_api_base,
        openai_api_key   = settings.openai_api_key,
        openai_model     = settings.openai_model,
        embedding_model  = settings.embedding_model,
    )

    evaluator = RAGASEvaluator(retriever, reranker, engine, top_k=top_k)
    report    = evaluator.evaluate(benchmark_data)
    report.benchmark_path = str(benchmark)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), indent=2))
    console.print(f"\n[dim]Report written → {output}[/dim]")
    _print_report(report)


if __name__ == "__main__":
    app()
