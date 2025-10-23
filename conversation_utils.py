import logging
from homeassistant.components import conversation
from homeassistant.helpers import intent

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


async def abort_response(user_input: conversation.ConversationInput):
    """Abort the current multi-turn flow politely."""
    _LOGGER.debug("Abort requested for input=%s", user_input.text)
    return await make_response("Okay, abgebrochen.", user_input, end=True)


async def error_response(user_input: conversation.ConversationInput, msg: str = None):
    """Return a standardized error response."""
    message = msg or "Entschuldigung, ich habe das nicht verstanden. Bitte wiederhole."
    _LOGGER.debug("Error response for input=%s → %s", user_input.text, message)
    return await make_response(message, user_input)


def with_new_text(user_input: conversation.ConversationInput, new_text: str) -> conversation.ConversationInput:
    """Clone a ConversationInput with modified text but same metadata."""
    return conversation.ConversationInput(
        text=new_text,
        context=user_input.context,
        conversation_id=user_input.conversation_id,
        device_id=user_input.device_id,
        satellite_id=getattr(user_input, "satellite_id", None),
        language=user_input.language,
        agent_id=getattr(user_input, "agent_id", None),
        extra_system_prompt=getattr(user_input, "extra_system_prompt", None),
    )
