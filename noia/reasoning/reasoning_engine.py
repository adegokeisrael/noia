"""
noia/reasoning/reasoning_engine.py
────────────────────────────────────
Central LLM reasoning core for NOIA.

Responsibilities:
  1. Select the appropriate structured prompt template (by task mode).
  2. Build the augmented prompt: context block + engineer query.
  3. Call the configured LLM backend (local llama.cpp or OpenAI-compatible API).
  4. Parse the structured JSON response.
  5. Invoke the groundedness checker.
  6. Return a fully annotated NOIAResponse object.

This module implements the core intelligence loop referenced in the FG-AINN
submission §7.b code snippet.

Supported task modes:
  - "query"            → General NOC runbook/knowledge query  (TC-01)
  - "rca"              → Outage root cause analysis            (TC-02)
  - "summarise"        → Incident thread summarisation         (TC-03)
  - "recommend"        → Fix recommendation                    (TC-04)
  - "generate_article" → KB article generation                 (TC-05)
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from noia.reasoning.groundedness_checker import GroundednessChecker, GroundednessResult
from noia.reasoning.prompt_templates import build_context_block, get_template
from noia.retrieval.hybrid_retriever import RetrievedChunk

logger = logging.getLogger(__name__)

TASK_MODES = frozenset(["query", "rca", "recommend", "summarise", "generate_article"])


# ── Response data class ───────────────────────────────────────────────────────

@dataclass
class NOIAResponse:
    """
    Fully annotated response returned by the reasoning engine for all task modes.
    """

    mode: str
    query: str
    parsed: dict[str, Any]              # structured LLM output
    raw_text: str                        # full raw LLM response text
    context_chunks: list[dict]           # chunks used for generation
    groundedness: GroundednessResult     # groundedness assessment
    response_time_ms: float              # end-to-end latency
    llm_backend: str                     # "local" | "openai"
    model_name: str
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "mode":               self.mode,
            "query":              self.query,
            "response":           self.parsed,
            "groundedness":       self.groundedness.to_dict(),
            "review_required":    self.groundedness.review_required,
            "response_time_ms":   round(self.response_time_ms, 1),
            "llm_backend":        self.llm_backend,
            "model":              self.model_name,
            "context_sources":    [
                {
                    "source_num":  i + 1,
                    "doc_id":      c.get("doc_id", ""),
                    "title":       c.get("title", ""),
                    "source_type": c.get("source_type", ""),
                    "timestamp":   c.get("timestamp", "")[:10],
                    "score":       round(c.get("score", 0.0), 4),
                }
                for i, c in enumerate(self.context_chunks)
            ],
            "errors":             self.errors,
        }


# ── LLM client wrappers ───────────────────────────────────────────────────────

class _LocalLLMClient:
    """
    Thin wrapper around llama-cpp-python for local GGUF model inference.
    """

    def __init__(self, model_path: str, n_ctx: int = 4096, n_threads: int = 4) -> None:
        from llama_cpp import Llama  # noqa: PLC0415
        logger.info("Loading local LLM from %s (n_ctx=%d).", model_path, n_ctx)
        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
        )
        self.model_name = model_path.split("/")[-1]

    def complete(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.1) -> str:
        output = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|im_end|>", "</s>", "[INST]"],
            echo=False,
        )
        return output["choices"][0]["text"].strip()


class _OpenAILLMClient:
    """
    Thin wrapper around the OpenAI-compatible REST API.
    """

    def __init__(self, api_base: str, api_key: str, model: str) -> None:
        from openai import OpenAI  # noqa: PLC0415
        self._client = OpenAI(base_url=api_base, api_key=api_key or "none")
        self.model_name = model

    def complete(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.1) -> str:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()


# ── JSON parsing helper ───────────────────────────────────────────────────────

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _parse_json_response(text: str, mode: str) -> tuple[dict, list[str]]:
    """
    Attempt to parse a JSON object from the LLM response text.

    Returns (parsed_dict, errors). On failure, returns a fallback dict.
    """
    errors: list[str] = []

    # 1. Try to extract from a JSON code fence
    fence_match = _JSON_FENCE_RE.search(text)
    candidate   = fence_match.group(1) if fence_match else text

    # 2. Find the outermost JSON object
    brace_start = candidate.find("{")
    brace_end   = candidate.rfind("}") + 1
    if brace_start != -1 and brace_end > brace_start:
        candidate = candidate[brace_start:brace_end]

    try:
        return json.loads(candidate), errors
    except json.JSONDecodeError as exc:
        errors.append(f"JSON parse error: {exc}")
        logger.warning("JSON parse failed for mode=%s: %s", mode, exc)

    # 3. Fallback: return raw text wrapped in a generic dict
    return {"raw_response": text, "parse_error": str(errors[0])}, errors


# ── Reasoning engine ──────────────────────────────────────────────────────────

class ReasoningEngine:
    """
    Orchestrates prompt construction, LLM inference, response parsing,
    and groundedness checking for all NOIA task modes.

    Parameters
    ----------
    llm_backend : str
        "local" (llama.cpp) or "openai" (OpenAI-compatible API).
    local_model_path : str
        Path to GGUF model file (used if llm_backend="local").
    openai_api_base : str
        Base URL for OpenAI-compatible API.
    openai_api_key : str
        API key.
    openai_model : str
        Model name for the OpenAI-compatible API.
    embedding_model : str
        Sentence-transformers model for groundedness checking.
    groundedness_theta : float
        Per-claim similarity threshold θ.
    groundedness_review_threshold : float
        G(r) below this triggers REVIEW_REQUIRED.
    max_tokens : int
        Maximum token budget for LLM completion.
    temperature : float
        LLM sampling temperature (low = more deterministic).
    """

    def __init__(
        self,
        llm_backend: str                 = "local",
        local_model_path: str            = "models/mistral-7b-instruct-v0.3.Q4_K_M.gguf",
        openai_api_base: str             = "https://api.openai.com/v1",
        openai_api_key: str              = "",
        openai_model: str                = "gpt-4o-mini",
        embedding_model: str             = "sentence-transformers/all-MiniLM-L6-v2",
        groundedness_theta: float        = 0.78,
        groundedness_review_threshold: float = 0.70,
        max_tokens: int                  = 1024,
        temperature: float               = 0.1,
    ) -> None:
        self.llm_backend    = llm_backend
        self.max_tokens     = max_tokens
        self.temperature    = temperature

        self._llm_client    = None
        self._llm_init_args = {
            "local": dict(model_path=local_model_path),
            "openai": dict(api_base=openai_api_base, api_key=openai_api_key, model=openai_model),
        }

        self._grounder = GroundednessChecker(
            embedding_model  = embedding_model,
            theta            = groundedness_theta,
            review_threshold = groundedness_review_threshold,
        )

    # ── LLM client (lazy) ─────────────────────────────────────────────────────

    def _get_llm(self):
        if self._llm_client is None:
            args = self._llm_init_args[self.llm_backend]
            if self.llm_backend == "local":
                self._llm_client = _LocalLLMClient(**args)
            else:
                self._llm_client = _OpenAILLMClient(**args)
        return self._llm_client

    # ── Prompt builder ────────────────────────────────────────────────────────

    def _build_prompt(self, mode: str, query: str, chunks: list[RetrievedChunk]) -> str:
        """Construct the augmented prompt from template + context + query."""
        template = get_template(mode)
        context  = build_context_block([c.to_dict() for c in chunks])
        return template.format(context=context, query=query)

    # ── Core reasoning loop ───────────────────────────────────────────────────

    def reason(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        mode: str = "query",
    ) -> NOIAResponse:
        """
        Execute the full reasoning pipeline for a given query and context.

        Parameters
        ----------
        query : str
            Engineer's natural-language input (question, incident description,
            thread, fault notification, etc.).
        chunks : list[RetrievedChunk]
            Retrieved context chunks from the hybrid retriever + reranker.
        mode : str
            Task mode: 'query' | 'rca' | 'recommend' | 'summarise' |
            'generate_article'.

        Returns
        -------
        NOIAResponse
            Fully annotated response with parsed output, citations, and
            groundedness assessment.
        """
        if mode not in TASK_MODES:
            raise ValueError(f"Unknown mode '{mode}'. Valid: {sorted(TASK_MODES)}")

        t_start = time.perf_counter()
        errors:  list[str] = []

        # 1. Build prompt
        prompt = self._build_prompt(mode, query, chunks)
        logger.debug("Prompt length: %d chars | mode=%s", len(prompt), mode)

        # 2. Call LLM
        llm = self._get_llm()
        try:
            raw_text = llm.complete(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception as exc:  # noqa: BLE001
            msg = f"LLM inference error: {exc}"
            logger.error(msg)
            errors.append(msg)
            raw_text = '{"error": "LLM inference failed."}'

        # 3. Parse structured output
        parsed, parse_errors = _parse_json_response(raw_text, mode)
        errors.extend(parse_errors)

        # 4. Groundedness check
        chunk_dicts = [c.to_dict() for c in chunks]
        groundedness = self._grounder.check(raw_text, chunk_dicts)

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        logger.info(
            "Reasoning complete | mode=%s | G(r)=%.3f | review=%s | %.0fms",
            mode, groundedness.score, groundedness.review_required, elapsed_ms,
        )

        return NOIAResponse(
            mode            = mode,
            query           = query,
            parsed          = parsed,
            raw_text        = raw_text,
            context_chunks  = chunk_dicts,
            groundedness    = groundedness,
            response_time_ms= elapsed_ms,
            llm_backend     = self.llm_backend,
            model_name      = getattr(llm, "model_name", "unknown"),
            errors          = errors,
        )

    # ── Convenience wrappers per TC ───────────────────────────────────────────

    def answer_query(self, query: str, chunks: list[RetrievedChunk]) -> NOIAResponse:
        """TC-01: Answer a general NOC runbook query."""
        return self.reason(query, chunks, mode="query")

    def analyse_rca(self, incident_description: str, chunks: list[RetrievedChunk]) -> NOIAResponse:
        """TC-02: Perform root cause analysis."""
        return self.reason(incident_description, chunks, mode="rca")

    def summarise_incident(self, incident_thread: str, chunks: list[RetrievedChunk]) -> NOIAResponse:
        """TC-03: Summarise an incident thread."""
        return self.reason(incident_thread, chunks, mode="summarise")

    def recommend_fix(self, fault_description: str, chunks: list[RetrievedChunk]) -> NOIAResponse:
        """TC-04: Generate ranked fix recommendations."""
        return self.reason(fault_description, chunks, mode="recommend")

    def generate_kb_article(self, resolved_incident: str, chunks: list[RetrievedChunk]) -> NOIAResponse:
        """TC-05: Auto-generate a KB article from a resolved incident."""
        return self.reason(resolved_incident, chunks, mode="generate_article")
