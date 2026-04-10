"""Integration tests for domain filtering with real LLM (requires Ollama)."""

import pytest
from unittest.mock import MagicMock
from homeassistant.components import conversation


@pytest.fixture
def hass():
    """Mock Home Assistant with area containing multiple entity types."""
    return MagicMock()


def make_input(text: str):
    return conversation.ConversationInput(
        text=text, context=MagicMock(), conversation_id="test_id",
        device_id="test_device", language="de",
    )


# Synonym sets for lenient assertion
_BRIGHT_UP = {"heller", "erhöhen", "aufdrehen", "hochdrehen", "aufhellen", "beleuchtung erhöhen", "licht heller"}
_BRIGHT_DOWN = {"dunkler", "dimmen", "reduzieren", "runterdrehen", "abdunkeln", "beleuchtung reduzieren"}
_LIGHT_SYNONYMS = {"licht", "beleuchtung", "lampe", "lampen"}


def _contains_any(text: str, words: set) -> bool:
    text = text.lower()
    return any(w in text for w in words)


class TestDomainFilteringLLM:
    """LLM-dependent domain filtering tests."""

    @pytest.mark.asyncio
    async def test_keyword_intent_returns_domain(self, hass, integration_llm_config):
        from multistage_assist.capabilities.keyword_intent import KeywordIntentCapability
        cap = KeywordIntentCapability(hass, integration_llm_config)
        result = await cap.run(make_input("Mache das Licht im Wohnzimmer heller"))
        assert result.get("domain") == "light"
        assert result.get("intent") == "HassLightSet"

    @pytest.mark.asyncio
    async def test_brightness_command_only_returns_lights(self, hass, integration_llm_config):
        from multistage_assist.capabilities.keyword_intent import KeywordIntentCapability
        cap = KeywordIntentCapability(hass, integration_llm_config)
        result = await cap.run(make_input("Mache das Licht im Wohnzimmer heller"))
        assert result.get("intent") == "HassLightSet"
        assert result.get("domain") == "light"
        slots = result.get("slots", {})
        assert "wohnzimmer" in slots.get("area", "").lower()

    @pytest.mark.asyncio
    async def test_implicit_intent_transforms_zu_dunkel(self, hass, integration_llm_config):
        from multistage_assist.capabilities.implicit_intent import ImplicitIntentCapability
        cap = ImplicitIntentCapability(hass, integration_llm_config)
        result = await cap.run(make_input("im Wohnzimmer ist es zu dunkel"))
        assert isinstance(result, list) and len(result) == 1
        command = result[0].lower()
        assert "wohnzimmer" in command
        assert _contains_any(command, _BRIGHT_UP | _LIGHT_SYNONYMS), \
            f"Expected brightness-up or light synonym, got: {result[0]}"

    @pytest.mark.asyncio
    async def test_full_flow_zu_dunkel(self, hass, integration_llm_config):
        from multistage_assist.capabilities.implicit_intent import ImplicitIntentCapability
        from multistage_assist.capabilities.keyword_intent import KeywordIntentCapability
        implicit_cap = ImplicitIntentCapability(hass, integration_llm_config)
        clarified = await implicit_cap.run(make_input("im Wohnzimmer ist es zu dunkel"))
        assert isinstance(clarified, list) and len(clarified) > 0
        keyword_cap = KeywordIntentCapability(hass, integration_llm_config)
        ki_result = await keyword_cap.run(make_input(clarified[0]))
        assert ki_result.get("domain") == "light"
        assert ki_result.get("intent") == "HassLightSet"


class TestImplicitCommandsLLM:
    """LLM-dependent implicit command transformation tests."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("input_text,expected_words", [
        ("im Wohnzimmer ist es zu dunkel", _BRIGHT_UP | _LIGHT_SYNONYMS),
        ("es ist zu dunkel hier", _BRIGHT_UP | _LIGHT_SYNONYMS),
        ("zu hell im Bad", _BRIGHT_DOWN | _LIGHT_SYNONYMS),
        ("es ist zu hell", _BRIGHT_DOWN | _LIGHT_SYNONYMS),
        ("im Büro ist es zu dunkel", _BRIGHT_UP | _LIGHT_SYNONYMS),
    ])
    async def test_implicit_brightness_transformation(self, hass, input_text, expected_words, integration_llm_config):
        from multistage_assist.capabilities.implicit_intent import ImplicitIntentCapability
        cap = ImplicitIntentCapability(hass, integration_llm_config)
        result = await cap.run(make_input(input_text))
        assert isinstance(result, list) and len(result) > 0
        combined = " ".join(result).lower()
        assert _contains_any(combined, expected_words), \
            f"Expected one of {expected_words} in '{combined}'"
