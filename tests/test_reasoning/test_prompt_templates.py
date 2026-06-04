"""tests/test_reasoning/test_prompt_templates.py"""

import pytest
from noia.reasoning.prompt_templates import (
    PROMPT_TEMPLATES,
    build_context_block,
    get_template,
)

VALID_MODES = ["query", "rca", "recommend", "summarise", "generate_article"]


class TestGetTemplate:
    @pytest.mark.parametrize("mode", VALID_MODES)
    def test_valid_mode_returns_string(self, mode):
        tpl = get_template(mode)
        assert isinstance(tpl, str)
        assert len(tpl) > 100

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown task mode"):
            get_template("invalid_mode")

    @pytest.mark.parametrize("mode", VALID_MODES)
    def test_template_contains_context_placeholder(self, mode):
        tpl = get_template(mode)
        assert "{context}" in tpl

    @pytest.mark.parametrize("mode", VALID_MODES)
    def test_template_contains_query_placeholder(self, mode):
        tpl = get_template(mode)
        assert "{query}" in tpl

    @pytest.mark.parametrize("mode", VALID_MODES)
    def test_template_contains_json_output_instruction(self, mode):
        tpl = get_template(mode)
        assert "JSON" in tpl or "json" in tpl

    @pytest.mark.parametrize("mode", VALID_MODES)
    def test_template_can_be_formatted(self, mode):
        tpl = get_template(mode)
        result = tpl.format(context="[SOURCE 1]\nTest context.", query="Test query.")
        assert "Test context." in result
        assert "Test query." in result


class TestBuildContextBlock:
    def test_basic_formatting(self):
        chunks = [
            {"text": "BGP recovery steps.", "title": "BGP Runbook",
             "source_type": "runbook", "timestamp": "2024-01-01T00:00:00"},
        ]
        block = build_context_block(chunks)
        assert "[Source 1" in block
        assert "BGP recovery steps." in block
        assert "RUNBOOK" in block

    def test_multiple_sources_numbered(self):
        chunks = [
            {"text": "Runbook text.", "title": "RB1",
             "source_type": "runbook", "timestamp": "2024-01-01"},
            {"text": "Incident text.", "title": "INC1",
             "source_type": "incident", "timestamp": "2024-02-01"},
        ]
        block = build_context_block(chunks)
        assert "[Source 1" in block
        assert "[Source 2" in block

    def test_empty_chunks_returns_block(self):
        block = build_context_block([])
        assert isinstance(block, str)

    def test_separator_between_sources(self):
        chunks = [
            {"text": "A.", "title": "T1", "source_type": "sla", "timestamp": "2024"},
            {"text": "B.", "title": "T2", "source_type": "sla", "timestamp": "2024"},
        ]
        block = build_context_block(chunks)
        assert "─" in block  # separator line
