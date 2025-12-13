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
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return f"{', '.join(names[:-1])} und {names[-1]}"


def normalize_speech_for_tts(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"(\d+)\.(\d+)", r"\1,\2", text)
    replacements = {
        "°C": " Grad Celsius",
        "°": " Grad",
        "%": " Prozent",
        "kWh": " Kilowattstunden",
        "kW": " Kilowatt",
        "W": " Watt",
        "V": " Volt",
        "A": " Ampere",
        "lx": " Lux",
        "lm": " Lumen",
    }
    for sym, spoken in replacements.items():
        text = re.sub(rf"{re.escape(sym)}(?=$|\s|[.,!?])", spoken, text)
    return text.strip()


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
    """Parse duration string/int to seconds."""
    if not duration:
        return 0
    try:
        return int(duration)
    except (ValueError, TypeError):
        pass
    if isinstance(duration, str):
        text = duration.lower()
        m_min = re.search(r"(\d+)\s*(m|min|minute)", text)
        m_sec = re.search(r"(\d+)\s*(s|sec|sekunde)", text)
        m_hr = re.search(r"(\d+)\s*(h|std|stunde)", text)
        total = 0
        if m_hr:
            total += int(m_hr.group(1)) * 3600
        if m_min:
            total += int(m_min.group(1)) * 60
        if m_sec:
            total += int(m_sec.group(1))
        if total == 0 and text.isdigit():
            return int(text) * 60
        return total
    return 0


def format_seconds_to_string(seconds: int) -> str:
    if seconds >= 3600:
        return f"{seconds/3600:.1f} Stunden"
    if seconds >= 60:
        return f"{int(seconds/60)} Minuten"
    return f"{seconds} Sekunden"


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
