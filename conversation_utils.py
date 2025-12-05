import logging
from homeassistant.components import conversation
from homeassistant.helpers import intent
from typing import List, Dict, Any

_LOGGER = logging.getLogger(__name__)

async def make_response(message: str, user_input: conversation.ConversationInput, end: bool = False):
    """Create a conversation response with a spoken message."""
    resp = intent.IntentResponse(language=user_input.language or "de")
    resp.response_type = intent.IntentResponseType.QUERY_ANSWER
    resp.async_set_speech(message)
    return conversation.ConversationResult(
        response=resp,
        conversation_id=user_input.conversation_id,
        continue_conversation=not end,
    )

async def error_response(user_input: conversation.ConversationInput, msg: str = None):
    message = msg or "Entschuldigung, ich habe das nicht verstanden."
    return await make_response(message, user_input)

def with_new_text(user_input: conversation.ConversationInput, new_text: str) -> conversation.ConversationInput:
    """Clone a ConversationInput with modified text but same metadata."""
    # Depending on HA version, satellite_id might be required positional arg
    # We fetch it safely, defaulting to None if missing on the input object
    satellite_id = getattr(user_input, "satellite_id", None)
    
    return conversation.ConversationInput(
        text=new_text,
        context=user_input.context,
        conversation_id=user_input.conversation_id,
        device_id=user_input.device_id,
        language=user_input.language,
        agent_id=getattr(user_input, "agent_id", None),
        satellite_id=satellite_id, # Pass explicitly
    )

# --- SHARED KEYWORDS ---
LIGHT_KEYWORDS: Dict[str, str] = {
    "licht": "lichter",
    "lampe": "lampen",
    "leuchte": "leuchten",
    "beleuchtung": "beleuchtungen",
    "spot": "spots",
}

COVER_KEYWORDS: Dict[str, str] = {
    "rollladen": "rollläden",
    "rollo": "rollos",
    "jalousie": "jalousien",
    "markise": "markisen",
    "beschattung": "beschattungen",
}

SENSOR_KEYWORDS: Dict[str, str] = {
    "temperatur": "temperaturen",
    "luftfeuchtigkeit": "luftfeuchtigkeiten",
    "feuchtigkeit": "feuchtigkeiten",
    "wert": "werte",
    "status": "status",
    "zustand": "zustände",
}

CLIMATE_KEYWORDS: Dict[str, str] = {
    "heizung": "heizungen",
    "thermostat": "thermostate",
    "klimaanlage": "klimaanlagen",
}

OTHER_ENTITY_PLURALS: Dict[str, str] = {
    "das fenster": "die fenster",
    "tür": "türen",
    "tor": "tore",
    "steckdose": "steckdosen",
    "ventilator": "ventilatoren",
    "gerät": "geräte",
}

_ENTITY_PLURALS: Dict[str, str] = {
    **LIGHT_KEYWORDS,
    **COVER_KEYWORDS,
    **SENSOR_KEYWORDS,
    **CLIMATE_KEYWORDS,
    **OTHER_ENTITY_PLURALS,
}

# --- HELPER FUNCTIONS ---

def join_names(names: List[str]) -> str:
    if not names: return ""
    if len(names) == 1: return names[0]
    return f"{', '.join(names[:-1])} und {names[-1]}"

def normalize_speech_for_tts(text: str) -> str:
    if not text: return ""
    import re
    text = re.sub(r"(\d+)\.(\d+)", r"\1,\2", text)
    replacements = {"°C": " Grad Celsius", "%": " Prozent", "kWh": " Kilowattstunden", "lx": " Lux"}
    for symbol, spoken in replacements.items():
        text = re.sub(rf"{re.escape(symbol)}(?=$|\s|[.,!?])", spoken, text)
    return text.strip()

def format_chat_history(history: List[Dict[str, str]], max_words: int = 500) -> str:
    full_text = []
    word_count = 0
    for turn in reversed(history):
        role = "User" if turn["role"] == "user" else "Jarvis"
        content = turn["content"]
        count = len(content.split())
        if word_count + count > max_words: break
        full_text.insert(0, f"{role}: {content}")
        word_count += count
    return "\n".join(full_text)

def filter_candidates_by_state(hass, entity_ids: List[str], intent_name: str) -> List[str]:
    if intent_name not in ("HassTurnOn", "HassTurnOff"): return entity_ids
    filtered = []
    for eid in entity_ids:
        st = hass.states.get(eid)
        if not st or st.state in ("unavailable", "unknown"): continue
        state = st.state
        domain = eid.split(".", 1)[0]
        keep = False
        if intent_name == "HassTurnOff":
            keep = (state != "closed") if domain == "cover" else (state != "off")
        elif intent_name == "HassTurnOn":
            keep = (state != "open") if domain == "cover" else (state != "on")
        if keep: filtered.append(eid)
    return filtered