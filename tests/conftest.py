import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

# Add the parent directory to sys.path to allow importing the integration as a package
# This assumes the current working directory is the integration root
sys.path.append(str(Path(__file__).parent.parent.parent))

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ConversationInput:
    text: str
    context: Any
    conversation_id: str | None
    device_id: str | None
    language: str | None
    agent_id: str | None = None
    satellite_id: str | None = None


@dataclass
class ConversationResult:
    response: Any
    conversation_id: str | None = None
    continue_conversation: bool = False


class IntentResponse:
    def __init__(self, language: str = "en"):
        self.language = language
        self.speech: dict[str, dict[str, Any]] = {}
        self.card: dict[str, Any] = {}
        self.error_code: str | None = None
        self.intent: dict[str, Any] | None = None
        self.matched_states: list[Any] = []
        self.unmatched_states: list[Any] = []
        self.async_set_speech = MagicMock(side_effect=self._set_speech)
        self.async_set_card = MagicMock()
        self.async_set_error = MagicMock()

    def _set_speech(self, speech, type="plain", extra_data=None):
        self.speech[type] = {"speech": speech, "extra_data": extra_data}


# Mock homeassistant package if not installed
if "homeassistant" not in sys.modules:
    ha_mock = MagicMock()
    sys.modules["homeassistant"] = ha_mock
    sys.modules["homeassistant.core"] = MagicMock()
    sys.modules["homeassistant.config_entries"] = MagicMock()
    sys.modules["homeassistant.const"] = MagicMock()
    sys.modules["homeassistant.helpers"] = MagicMock()
    sys.modules["homeassistant.helpers.typing"] = MagicMock()
    # Mock google package properly
    sys.modules["google"] = MagicMock()
    sys.modules["google"].__path__ = []  # Mark as package
    sys.modules["google.generativeai"] = MagicMock()

    # Mock google.genai.Client
    mock_genai = MagicMock()
    mock_client = MagicMock()
    mock_aio = MagicMock()
    mock_models = MagicMock()

    async def _mock_generate_content(*args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.text = '{"intent": "HassTurnOn", "slots": {"area": "Küche", "command": "an"}}'  # Default JSON
        return mock_resp

    mock_models.generate_content = AsyncMock(side_effect=_mock_generate_content)
    mock_aio.models = mock_models
    mock_client.aio = mock_aio
    mock_genai.Client = MagicMock(return_value=mock_client)

    mock_genai.Client = MagicMock(return_value=mock_client)

    sys.modules["google.genai"] = mock_genai
    sys.modules["google"].genai = mock_genai  # Ensure attribute access works

    try:
        import homeassistant.helpers.intent as intent

        sys.modules["homeassistant.helpers.intent"] = intent
    except ImportError:
        mock_intent = MagicMock()

        async def _mock_handle(*args, **kwargs):
            # The original mock_handle returned IntentResponse(language="de")
            # The instruction seems to be a partial snippet for a different function or a mix-up.
            # Reverting to original mock_handle behavior, but adding debug prints as requested.
            # The instruction's `intent` variable is not defined in this scope,
            # so we use the local `IntentResponse` class.
            print(f"DEBUG: _mock_handle called with args={args}, kwargs={kwargs}")
            user_input = kwargs.get("user_input")
            language = "de"
            if user_input and hasattr(user_input, "language") and user_input.language:
                language = user_input.language

            resp = IntentResponse(language=language)
            print(f"DEBUG: make_response created resp: {resp}, type: {type(resp)}")
            print(f"DEBUG: resp.speech: {resp.speech}, type: {type(resp.speech)}")
            # The instruction's `message` and `intent.IntentResponseType.QUERY_ANSWER` are not defined here.
            # Assuming the intent was to return a basic IntentResponse.
            return resp

        mock_intent.async_handle = AsyncMock(side_effect=_mock_handle)
        mock_intent.IntentResponse = IntentResponse
        mock_intent.IntentResponseType = MagicMock()
        mock_intent.IntentResponseType.QUERY_ANSWER = "query_answer"
        mock_intent.IntentResponseType.ACTION_DONE = "action_done"
        sys.modules["homeassistant.helpers.intent"] = mock_intent
        sys.modules["homeassistant.helpers"].intent = mock_intent  # Link to helpers
    sys.modules["homeassistant.components"] = MagicMock()
    sys.modules["homeassistant.components.conversation"] = MagicMock()
    sys.modules["homeassistant.components.conversation.default_agent"] = MagicMock()
    sys.modules["homeassistant.components.homeassistant"] = MagicMock()
    sys.modules["homeassistant.components.homeassistant.exposed_entities"] = MagicMock()
    sys.modules["homeassistant.data_entry_flow"] = MagicMock()
    sys.modules["homeassistant.data_entry_flow"].FlowResultType.FORM = "form"
    sys.modules["homeassistant.data_entry_flow"].FlowResultType.CREATE_ENTRY = (
        "create_entry"
    )
    # Mock Storage
    mock_storage = MagicMock()

    # Ensure Store class is mocked correctly with awaitable methods
    class MockStore:
        def __init__(self, *args, **kwargs):
            pass

        async def async_load(self):
            return {}

        async def async_save(self, data):
            pass

    mock_storage.Store = MockStore
    sys.modules["homeassistant.helpers.storage"] = mock_storage
    sys.modules["homeassistant.helpers.entity_registry"] = MagicMock()
    sys.modules["homeassistant.helpers.area_registry"] = MagicMock()
    sys.modules["homeassistant.helpers.device_registry"] = MagicMock()
    sys.modules["homeassistant.helpers.floor_registry"] = MagicMock()

    # Define specific constants or classes if needed by imports
    sys.modules["homeassistant.const"].CONF_PLATFORM = "platform"
    sys.modules["homeassistant.const"].DOMAIN = "multistage_assist"

    # We need to make sure 'from homeassistant.core import HomeAssistant' works
    # So we need to set attributes on the modules
    sys.modules["homeassistant.core"].HomeAssistant = MagicMock
    sys.modules["homeassistant.core"].callback = lambda x: x

    # Make ConfigFlow a class so inheritance works
    class MockConfigFlow:
        def __init__(self):
            self.hass = None
            self.context = {}

        def __init_subclass__(cls, **kwargs):
            pass

        async def async_step_user(self, user_input=None):
            return self.async_show_form(step_id="user")

        def async_show_form(self, **kwargs):
            return {"type": "form", "step_id": kwargs.get("step_id")}

        def async_create_entry(self, **kwargs):
            return {"type": "create_entry", **kwargs}

    sys.modules["homeassistant.config_entries"].ConfigFlow = MockConfigFlow
    sys.modules["homeassistant.config_entries"].ConfigEntry = MagicMock
    sys.modules["homeassistant.config_entries"].OptionsFlow = MagicMock
    sys.modules["homeassistant.config_entries"].SOURCE_USER = "user"

    # Fix for conversation.ConversationInput and Result
    # Moved to top of file

    sys.modules["homeassistant.components.conversation"].ConversationInput = (
        ConversationInput
    )
    sys.modules["homeassistant.components.conversation"].ConversationResult = (
        ConversationResult
    )
    sys.modules["homeassistant.components.conversation"].HOME_ASSISTANT_AGENT = (
        "homeassistant"
    )
    sys.modules["homeassistant.components.conversation"].async_converse = AsyncMock()
    sys.modules["homeassistant.components.conversation"].async_set_agent = MagicMock()
    sys.modules["homeassistant.components.conversation"].async_unset_agent = MagicMock()
    sys.modules["homeassistant.components.conversation"].intent = (
        mock_intent  # Link intent to conversation module
    )

    class MockAgent:
        @property
        def supported_languages(self):
            return []

        async def async_process(self, user_input):
            pass

    sys.modules["homeassistant.components.conversation"].AbstractConversationAgent = (
        MockAgent
    )

    # Fix for intent
    # Fix for intent
    # IntentResponse moved to top of file

    sys.modules["homeassistant.helpers.intent"].IntentResponse = IntentResponse

    # Mock custom_components
    sys.modules["custom_components"] = MagicMock()
    sys.modules["custom_components.multistage_assist"] = MagicMock()

    # We need to make sure custom_components.multistage_assist.conversation_utils points to the real one
    # But we can't import it yet because it might import other things.
    # However, conversation_utils seems to only import standard libs or HA stuff (which is mocked).
    # So let's try to import it via the package name we set up in sys.path
    # Link submodules to parents so 'from parent import child' works
    sys.modules["homeassistant"].core = sys.modules["homeassistant.core"]
    sys.modules["homeassistant"].config_entries = sys.modules[
        "homeassistant.config_entries"
    ]
    sys.modules["homeassistant"].const = sys.modules["homeassistant.const"]
    sys.modules["homeassistant"].helpers = sys.modules["homeassistant.helpers"]
    sys.modules["homeassistant"].components = sys.modules["homeassistant.components"]
    sys.modules["homeassistant"].data_entry_flow = sys.modules[
        "homeassistant.data_entry_flow"
    ]

    sys.modules["homeassistant.helpers"].typing = sys.modules[
        "homeassistant.helpers.typing"
    ]
    sys.modules["homeassistant.helpers"].intent = sys.modules[
        "homeassistant.helpers.intent"
    ]
    sys.modules["homeassistant.helpers"].storage = sys.modules[
        "homeassistant.helpers.storage"
    ]
    sys.modules["homeassistant.helpers"].entity_registry = sys.modules[
        "homeassistant.helpers.entity_registry"
    ]
    sys.modules["homeassistant.helpers"].area_registry = sys.modules[
        "homeassistant.helpers.area_registry"
    ]
    sys.modules["homeassistant.helpers"].device_registry = sys.modules[
        "homeassistant.helpers.device_registry"
    ]
    sys.modules["homeassistant.helpers"].floor_registry = sys.modules[
        "homeassistant.helpers.floor_registry"
    ]

    sys.modules["homeassistant.components"].conversation = sys.modules[
        "homeassistant.components.conversation"
    ]
    sys.modules["homeassistant.components"].homeassistant = sys.modules[
        "homeassistant.components.homeassistant"
    ]

    sys.modules["homeassistant.components.homeassistant"].exposed_entities = (
        sys.modules["homeassistant.components.homeassistant.exposed_entities"]
    )

    sys.modules["custom_components"].multistage_assist = sys.modules[
        "custom_components.multistage_assist"
    ]

    try:
        import multistage_assist.conversation_utils

        sys.modules["custom_components.multistage_assist.conversation_utils"] = (
            multistage_assist.conversation_utils
        )
        sys.modules["custom_components.multistage_assist"].conversation_utils = (
            multistage_assist.conversation_utils
        )
    except ImportError:
        sys.modules["custom_components.multistage_assist.conversation_utils"] = (
            MagicMock()
        )

    try:
        import multistage_assist.const

        sys.modules["custom_components.multistage_assist.const"] = (
            multistage_assist.const
        )
        sys.modules["custom_components.multistage_assist"].const = (
            multistage_assist.const
        )
    except ImportError:
        sys.modules["custom_components.multistage_assist.const"] = MagicMock()

import pytest

# Now we can import these safely (they will be mocks if HA is not installed)
from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_PLATFORM


"""Fixtures for testing."""


@pytest.fixture
def hass():
    """Mock Home Assistant."""
    hass = MagicMock(spec=HomeAssistant)
    hass.data = {}
    hass.config = MagicMock()
    hass.config.components = set()
    hass.config_entries = MagicMock()
    hass.config_entries.async_reload = AsyncMock()
    hass.async_add_executor_job = AsyncMock(
        side_effect=lambda f, *args, **kwargs: f(*args, **kwargs)
    )
    hass.states = MagicMock()
    hass.bus = MagicMock()
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock()

    # Mock notify services for timer capability
    def mock_async_services():
        return {
            "notify": {
                "mobile_app_sm_a566b": {"description": "SM-A566B"},
            }
        }

    hass.services.async_services = MagicMock(side_effect=mock_async_services)

    # Mock State Machine
    states = {}
    hass.states.get = lambda entity_id: states.get(entity_id)
    hass.states.async_all = lambda: list(states.values())
    hass.states.set = lambda entity_id, state, attributes=None: states.update(
        {
            entity_id: MagicMock(
                entity_id=entity_id,
                state=state,
                attributes=attributes or {},
                domain=entity_id.split(".")[0],
            )
        }
    )

    class MockArea:
        def __init__(self, id, name):
            self.id = id
            self.name = name

    class MockState:
        def __init__(self, entity_id, state, attributes):
            self.entity_id = entity_id
            self.state = state
            self.attributes = attributes

    # Populate areas
    areas = [
        MockArea(id="kitchen_id", name="Küche"),
        MockArea(id="utility_room_id", name="Hauswirtschaftsraum"),
        MockArea(id="garage_id", name="Garage"),
        MockArea(id="guest_bath_id", name="Gäste Badezimmer"),
    ]
    hass.data["area_registry"] = MagicMock()
    hass.data["area_registry"].async_create = MagicMock()
    for area in areas:
        hass.data["area_registry"].async_create(area.name)
        # We need to ensure async_list_areas returns these

    hass.data["area_registry"].async_list_areas = MagicMock(return_value=areas)

    # Populate entities
    # light.kuche (Name: "Küche")
    # light.kuche_spots (Name: "Küche Spots")
    # light.hauswirtschaftsraum
    # light.garage
    # light.gaste_badezimmer

    mock_states = [
        MockState("light.buro", "off", {"friendly_name": "Büro"}),
        MockState("light.kuche", "off", {"friendly_name": "Küche"}),
        MockState("light.kuche_spots", "off", {"friendly_name": "Küche Spots"}),
        MockState(
            "light.hauswirtschaftsraum", "off", {"friendly_name": "Hauswirtschaftsraum"}
        ),
        MockState("light.garage", "off", {"friendly_name": "Garage"}),
        MockState(
            "light.gaste_badezimmer", "on", {"friendly_name": "Gäste Badezimmer"}
        ),
        MockState("light.badezimmer", "on", {"friendly_name": "Badezimmer"}),
        MockState("light.dusche", "on", {"friendly_name": "Dusche"}),
        MockState(
            "light.badezimmer_spiegel", "on", {"friendly_name": "Badezimmer Spiegel"}
        ),
        MockState("cover.buro_nord", "closed", {"friendly_name": "Büro Nord"}),
        MockState("cover.buro_ost", "closed", {"friendly_name": "Büro Ost"}),
        MockState("script.temporary_control_generic", "off", {}),
    ]

    for state in mock_states:
        hass.states.set(state.entity_id, state.state, state.attributes)

    # Mock Registries
    mock_er = MagicMock()
    mock_er.entities = {}
    mock_er.async_get = MagicMock(return_value=None)  # Helper for individual entity

    class MockEntity:
        def __init__(
            self, entity_id, area_id, device_id, disabled_by, original_name, domain
        ):
            self.entity_id = entity_id
            self.area_id = area_id
            self.device_id = device_id
            self.disabled_by = disabled_by
            self.original_name = original_name
            self.domain = domain

    # Populate Entity Registry
    entities = [
        MockEntity(
            entity_id="light.buro",
            area_id="buro",
            device_id="dev_buro",
            disabled_by=None,
            original_name="Büro Licht",
            domain="light",
        ),
        MockEntity(
            entity_id="light.kuche",
            area_id="kuche",
            device_id="dev_kuche",
            disabled_by=None,
            original_name="Küche Licht",
            domain="light",
        ),
        MockEntity(
            entity_id="light.kuche_spots",
            area_id="kuche",
            device_id="dev_kuche_spots",
            disabled_by=None,
            original_name="Küche Spots",
            domain="light",
        ),
        MockEntity(
            entity_id="light.hauswirtschaftsraum",
            area_id="hwr",
            device_id="dev_hwr",
            disabled_by=None,
            original_name="Hauswirtschaftsraum Licht",
            domain="light",
        ),
        MockEntity(
            entity_id="light.garage",
            area_id="garage",
            device_id="dev_garage",
            disabled_by=None,
            original_name="Garage Licht",
            domain="light",
        ),
        MockEntity(
            entity_id="light.gaste_badezimmer",
            area_id="gastebad",
            device_id="dev_gastebad",
            disabled_by=None,
            original_name="Gäste Badezimmer Licht",
            domain="light",
        ),
        MockEntity(
            entity_id="light.badezimmer",
            area_id="badezimmer",
            device_id="dev_badezimmer",
            disabled_by=None,
            original_name="Badezimmer",
            domain="light",
        ),
        MockEntity(
            entity_id="light.dusche",
            area_id="badezimmer",
            device_id="dev_dusche",
            disabled_by=None,
            original_name="Dusche",
            domain="light",
        ),
        MockEntity(
            entity_id="light.badezimmer_spiegel",
            area_id="badezimmer",
            device_id="dev_badezimmer_spiegel",
            disabled_by=None,
            original_name="Badezimmer Spiegel",
            domain="light",
        ),
        MockEntity(
            entity_id="cover.buro_nord",
            area_id="buro",
            device_id="dev_buro_cover_nord",
            disabled_by=None,
            original_name="Büro Nord",
            domain="cover",
        ),
        MockEntity(
            entity_id="cover.buro_ost",
            area_id="buro",
            device_id="dev_buro_cover_ost",
            disabled_by=None,
            original_name="Büro Ost",
            domain="cover",
        ),
    ]
    mock_er.entities = {e.entity_id: e for e in entities}
    mock_er.async_get.side_effect = lambda eid: mock_er.entities.get(eid)

    sys.modules["homeassistant.helpers.entity_registry"].async_get = MagicMock(
        return_value=mock_er
    )

    # Mock Area Registry
    mock_ar = MagicMock()

    class MockArea:
        def __init__(self, id, name, floor_id):
            self.id = id
            self.name = name
            self.floor_id = floor_id

    areas = [
        MockArea(id="buro", name="Büro", floor_id=None),
        MockArea(id="kuche", name="Küche", floor_id=None),
        MockArea(id="hwr", name="Hauswirtschaftsraum", floor_id=None),
        MockArea(id="garage", name="Garage", floor_id=None),
        MockArea(id="gastebad", name="Gästebad", floor_id=None),
        MockArea(id="badezimmer", name="Badezimmer", floor_id=None),
    ]
    mock_ar.async_list_areas = MagicMock(return_value=areas)
    mock_ar.async_get_area = MagicMock(
        side_effect=lambda aid: next((a for a in areas if a.id == aid), None)
    )
    sys.modules["homeassistant.helpers.area_registry"].async_get = MagicMock(
        return_value=mock_ar
    )

    # Mock Device Registry
    mock_dr = MagicMock()

    class MockDevice:
        def __init__(self, id, area_id, name):
            self.id = id
            self.area_id = area_id
            self.name = name

    devices = [
        MockDevice(id="dev_buro", area_id="buro", name="Büro Device"),
        MockDevice(id="dev_kuche", area_id="kuche", name="Küche Device"),
        MockDevice(id="dev_kuche_spots", area_id="kuche", name="Küche Spots Device"),
        MockDevice(id="dev_hwr", area_id="hwr", name="HWR Device"),
        MockDevice(id="dev_garage", area_id="garage", name="Garage Device"),
        MockDevice(id="dev_gastebad", area_id="gastebad", name="Gästebad Device"),
    ]
    mock_dr.devices = {d.id: d for d in devices}
    mock_dr.async_get = MagicMock(side_effect=lambda did: mock_dr.devices.get(did))
    sys.modules["homeassistant.helpers.device_registry"].async_get = MagicMock(
        return_value=mock_dr
    )

    # Mock Floor Registry
    mock_fr = MagicMock()
    mock_fr.async_list_floors = MagicMock(return_value=[])
    sys.modules["homeassistant.helpers.floor_registry"].async_get = MagicMock(
        return_value=mock_fr
    )

    # Mock Exposed Entities
    sys.modules[
        "homeassistant.components.homeassistant.exposed_entities"
    ].async_should_expose = MagicMock(return_value=True)

    return hass


@pytest.fixture
def config_entry():
    """Mock Config Entry."""
    entry = MagicMock(spec=ConfigEntry)
    entry.entry_id = "test_entry_id"
    entry.domain = "multistage_assist"
    entry.data = {
        "stage1_ip": "127.0.0.1",
        "stage1_port": 5000,
        "stage1_model": "test_model",
        "google_api_key": "test_key",
        "stage2_model": "gemini-test",
    }
    entry.options = {}
    entry.add_update_listener = MagicMock()
    entry.async_on_unload = MagicMock()
    return entry


@pytest.fixture
def mock_conversation(hass):
    """Mock conversation component."""
    # Since we mocked the module in sys.modules, we can just return those mocks
    # But we need to make sure the code uses them.
    # The code imports 'conversation' from 'homeassistant.components'.
    # We mocked 'homeassistant.components.conversation'.

    mock_mod = sys.modules["homeassistant.components.conversation"]

    yield {
        "async_set_agent": mock_mod.async_set_agent,
        "async_unset_agent": mock_mod.async_unset_agent,
        "async_converse": mock_mod.async_converse,
    }
