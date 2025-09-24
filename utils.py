from homeassistant.helpers import intent
from homeassistant.components import conversation


def make_response(message: str, language: str = "de") -> conversation.ConversationResult:
    """Build a placeholder conversation response."""
    resp = intent.IntentResponse(language=language)
    resp.response_type = intent.IntentResponseType.QUERY_ANSWER
    resp.async_set_speech(message)
    return conversation.ConversationResult(response=resp)
