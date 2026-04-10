"""Tests for JSON parsing utilities (REQ-QUAL-001, REQ-TEST-002).

Covers: extract_json_from_llm_string — the core LLM output parser.
"""

import json
import pytest

from multistage_assist.utils.json_utils import extract_json_from_llm_string


class TestExtractJsonFromLlmString:
    """Tests for robust JSON extraction from LLM output."""

    def test_plain_json(self):
        result = extract_json_from_llm_string('{"intent": "HassTurnOn", "slots": {}}')
        assert result["intent"] == "HassTurnOn"

    def test_json_in_markdown_block(self):
        text = '```json\n{"intent": "HassTurnOff"}\n```'
        result = extract_json_from_llm_string(text)
        assert result["intent"] == "HassTurnOff"

    def test_json_in_markdown_block_no_language(self):
        text = '```\n{"key": "value"}\n```'
        result = extract_json_from_llm_string(text)
        assert result["key"] == "value"

    def test_json_with_preamble(self):
        text = 'Here is the result:\n{"tool": "list_entities", "args": {"domain": "light"}}'
        result = extract_json_from_llm_string(text)
        assert result["tool"] == "list_entities"

    def test_json_with_postamble(self):
        text = '{"final_answer": ["light.wohnzimmer"]}\nI found it.'
        result = extract_json_from_llm_string(text)
        assert result["final_answer"] == ["light.wohnzimmer"]

    def test_nested_json(self):
        data = {"intent": "HassLightSet", "slots": {"brightness": 50, "area": "Küche"}}
        result = extract_json_from_llm_string(json.dumps(data))
        assert result == data

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            extract_json_from_llm_string("not json at all")

    def test_empty_string_raises(self):
        with pytest.raises(json.JSONDecodeError):
            extract_json_from_llm_string("")

    def test_whitespace_around_json(self):
        result = extract_json_from_llm_string('  \n  {"key": "value"}  \n  ')
        assert result["key"] == "value"

    def test_german_text_in_values(self):
        text = '{"response": "Das Licht im Wohnzimmer ist eingeschaltet."}'
        result = extract_json_from_llm_string(text)
        assert "Wohnzimmer" in result["response"]

    def test_array_extraction(self):
        text = 'The implicit commands are: ["Mach das Licht an", "Mach die Heizung aus"]'
        result = extract_json_from_llm_string(text)
        assert result == ["Mach das Licht an", "Mach die Heizung aus"]
