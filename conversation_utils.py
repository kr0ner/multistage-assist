import logging
import re
from typing import List, Dict, Any, Optional

from homeassistant.core import HomeAssistant
from homeassistant.components import conversation
from homeassistant.helpers import intent

# Import keyword constants
from .constants.entity_keywords import (
    LIGHT_KEYWORDS,
    COVER_KEYWORDS,
    SWITCH_KEYWORDS,
    FAN_KEYWORDS,
    MEDIA_KEYWORDS,
    SENSOR_KEYWORDS,
    CLIMATE_KEYWORDS,
    VACUUM_KEYWORDS,
    TIMER_KEYWORDS,
    CALENDAR_KEYWORDS,
    AUTOMATION_KEYWORDS,
    OTHER_ENTITY_PLURALS,
    ENTITY_PLURALS as _ENTITY_PLURALS,
)

_LOGGER = logging.getLogger(__name__)

_PLURAL_CUES = {
    "alle",
    "sämtliche",
    "mehrere",
    "beide",
    "beiden",
    "viele",
    "verschiedene",
}
_NUM_WORDS = {
    "zwei",
    "drei",
    "vier",
    "fünf",
    "sechs",
    "sieben",
    "acht",
    "neun",
    "zehn",
    "elf",
    "zwölf",
}
_NUMERIC_PATTERN = re.compile(r"\b\d+\b")

# --- CONVERSATION HELPERS ---


async def make_response(
    message: str, user_input: conversation.ConversationInput, end: bool = False
) -> conversation.ConversationResult:
    """Create a conversation response."""
    resp = intent.IntentResponse(language=user_input.language or "de")
    resp.response_type = intent.IntentResponseType.QUERY_ANSWER
    resp.async_set_speech(message)
    return conversation.ConversationResult(
        response=resp,
        conversation_id=user_input.conversation_id,
        continue_conversation=not end,
    )


async def error_response(
    user_input: conversation.ConversationInput, msg: str = None
) -> conversation.ConversationResult:
    return await make_response(
        msg or "Entschuldigung, ich habe das nicht verstanden.", user_input
    )


def with_new_text(
    user_input: conversation.ConversationInput, new_text: str
) -> conversation.ConversationInput:
    """Clone input with new text."""
    satellite_id = getattr(user_input, "satellite_id", None)
    return conversation.ConversationInput(
        text=new_text,
        context=user_input.context,
        conversation_id=user_input.conversation_id,
        device_id=user_input.device_id,
        language=user_input.language,
        agent_id=getattr(user_input, "agent_id", None),
        satellite_id=satellite_id,
    )


# --- TEXT & STATE HELPERS ---


def join_names(names: List[str]) -> str:
    """Format list of names with German 'und' conjunction.
    
    This is a backward-compatible wrapper around response_builder.
    """
    from .utils.response_builder import join_names as _join_names
    return _join_names(names)


def normalize_speech_for_tts(text: str) -> str:
    """Normalize text for text-to-speech output.
    
    This is a backward-compatible wrapper around response_builder.
    """
    from .utils.response_builder import normalize_for_tts
    return normalize_for_tts(text)


def format_chat_history(history: List[Dict[str, str]], max_words: int = 500) -> str:
    full_text = []
    word_count = 0
    for turn in reversed(history):
        role = "User" if turn["role"] == "user" else "Jarvis"
        content = turn["content"]
        count = len(content.split())
        if word_count + count > max_words:
            break
        full_text.insert(0, f"{role}: {content}")
        word_count += count
    return "\n".join(full_text)


def parse_duration_string(duration: Any) -> int:
    """Parse duration string/int to seconds.
    
    This is a backward-compatible wrapper around the centralized duration utils.
    """
    from .utils.duration_utils import parse_german_duration
    return parse_german_duration(duration)


def format_seconds_to_string(seconds: int) -> str:
    """Format seconds to German duration string.
    
    This is a backward-compatible wrapper around the centralized duration utils.
    """
    from .utils.duration_utils import format_duration_simple
    return format_duration_simple(seconds)


def filter_candidates_by_state(
    hass: HomeAssistant, entity_ids: List[str], intent_name: str
) -> List[str]:
    """Filter entities based on intent (e.g. ignore ON lights for TurnOn)."""
    if intent_name not in ("HassTurnOn", "HassTurnOff"):
        return entity_ids
    filtered = []
    for eid in entity_ids:
        st = hass.states.get(eid)
        if not st or st.state in ("unavailable", "unknown"):
            continue
        state = st.state
        domain = eid.split(".", 1)[0]
        keep = False
        if intent_name == "HassTurnOff":
            keep = (state != "closed") if domain == "cover" else (state != "off")
        elif intent_name == "HassTurnOn":
            keep = (state != "open") if domain == "cover" else (state != "on")
        if keep:
            filtered.append(eid)
    return filtered
