"""MCP Tool capability with a polymorphic tool registry.

Provides tools for home discovery and personalized data management
to LLMs via the Model Context Protocol (MCP).
"""

import logging
import json
import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Type

from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
    floor_registry as fr,
)

from .base import Capability

# Import exposure check
try:
    from homeassistant.components.homeassistant.exposed_entities import async_should_expose
except ImportError:
    def async_should_expose(hass, conversation_agent_id, entity_id):
        return True

_LOGGER = logging.getLogger(__name__)


class McpTool(ABC):
    """Base class for all MCP tools."""

    name: str
    description: str
    parameters: Dict[str, Any]

    def __init__(self, capability: 'McpToolCapability'):
        self.cap = capability
        self.hass = capability.hass

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool logic."""
        pass

    def get_definition(self) -> Dict[str, Any]:
        """Return the tool definition for LLM prompts."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


# --- Tool Implementations ---

class ListAreasTool(McpTool):
    name = "list_areas"
    description = "List all rooms/areas configured in this Home Assistant smart home, including floor assignments and aliases."
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> Any:
        ar_reg = ar.async_get(self.hass)
        fr_reg = fr.async_get(self.hass)
        areas = []
        for area in ar_reg.areas.values():
            floor_name = None
            if area.floor_id:
                floor = fr_reg.async_get_floor(area.floor_id)
                if floor:
                    floor_name = floor.name
            areas.append({
                "id": area.id,
                "name": area.name,
                "floor": floor_name,
                "aliases": list(area.aliases) if area.aliases else [],
            })
        return areas


class ListEntitiesTool(McpTool):
    name = "list_entities"
    description = "List Home Assistant entities (smart devices) filtered by domain, area name, or device class. Domains include light, cover, switch, climate, sensor, media_player, fan, lock, vacuum."
    parameters = {
        "type": "object",
        "properties": {
            "domain": {"type": "string", "description": "Filter by domain (light, cover, switch, climate, sensor)"},
            "area_name": {"type": "string", "description": "Filter by area/room name"},
            "device_class": {"type": "string", "description": "Filter by device class (temperature, humidity, power)"},
            "limit": {"type": "integer", "description": "Max results (default 50)"}
        }
    }

    async def execute(self, domain=None, area_name=None, device_class=None, limit=50, **kwargs) -> Any:
        er_reg = er.async_get(self.hass)
        dr_reg = dr.async_get(self.hass)
        
        # Resolve area via centralized AreaResolver if provided
        target_area_ids = set()
        if area_name and self.cap.area_resolver:
            res = await self.cap.area_resolver.run(None, area_name=area_name, mode="area")
            match_name = res.get("match")
            if match_name:
                ar_reg = ar.async_get(self.hass)
                area = next((a for a in ar_reg.areas.values() if a.name == match_name), None)
                if area:
                    target_area_ids.add(area.id)
            
            if not target_area_ids:
                return []

        results = []
        for entry in er_reg.entities.values():
            if entry.disabled_by or not async_should_expose(self.hass, "conversation", entry.entity_id):
                continue
            if domain and entry.domain != domain:
                continue
            if target_area_ids:
                match = (entry.area_id in target_area_ids)
                if not match and entry.device_id:
                    device = dr_reg.async_get(entry.device_id)
                    match = (device and device.area_id in target_area_ids)
                if not match:
                    continue
            
            state = self.hass.states.get(entry.entity_id)
            if device_class:
                if not state or state.attributes.get("device_class") != device_class:
                    continue

            name = (state.attributes.get("friendly_name") if state else None) or entry.original_name or entry.name or entry.entity_id
            results.append({
                "entity_id": entry.entity_id,
                "name": name,
                "area_id": entry.area_id,
                "domain": entry.domain,
            })
            if len(results) >= limit:
                break
        return results


class GetEntityDetailsTool(McpTool):
    name = "get_entity_details"
    description = "Get detailed info and state about a specific entity."
    parameters = {
        "type": "object",
        "properties": {"entity_id": {"type": "string"}},
        "required": ["entity_id"]
    }

    async def execute(self, entity_id=None, **kwargs) -> Any:
        state = self.hass.states.get(entity_id)
        if not state or not async_should_expose(self.hass, "conversation", entity_id):
            return {"error": "Entity not found or not exposed"}
        
        er_reg = er.async_get(self.hass)
        entry = er_reg.async_get(entity_id)
        
        return {
            "entity_id": entity_id,
            "state": state.state,
            "attributes": dict(state.attributes),
            "area_id": entry.area_id if entry else None,
            "aliases": list(entry.aliases) if entry and entry.aliases else [],
        }


class ListAutomationsTool(McpTool):
    name = "list_automations"
    description = "List all automations exposed to voice assistants."
    parameters = {
        "type": "object",
        "properties": {"limit": {"type": "integer", "description": "Max results (default 50)"}}
    }

    async def execute(self, limit=50, **kwargs) -> Any:
        er_reg = er.async_get(self.hass)
        results = []
        automations = [e for e in er_reg.entities.values() if e.domain == "automation"]
        for entry in automations:
            if not async_should_expose(self.hass, "conversation", entry.entity_id):
                continue
            state = self.hass.states.get(entry.entity_id)
            name = (state.attributes.get("friendly_name") if state else None) or entry.original_name or entry.name or entry.entity_id
            results.append({
                "entity_id": entry.entity_id,
                "name": name,
                "state": state.state if state else "unknown"
            })
            if len(results) >= limit:
                break
        return results


class GetAutomationDetailsTool(McpTool):
    name = "get_automation_details"
    description = "Get details/config of a specific automation."
    parameters = {
        "type": "object",
        "properties": {"entity_id": {"type": "string"}},
        "required": ["entity_id"]
    }

    async def execute(self, entity_id=None, **kwargs) -> Any:
        state = self.hass.states.get(entity_id)
        if not state or not async_should_expose(self.hass, "conversation", entity_id):
            return {"error": "Automation not found or not exposed"}
        return {
            "entity_id": entity_id,
            "name": state.attributes.get("friendly_name", entity_id),
            "state": state.state,
            "attributes": dict(state.attributes)
        }


class StorePersonalDataTool(McpTool):
    name = "store_personal_data"
    description = "Store a personal fact about the household users for future reference (e.g., names, preferences, birthdays, roles)."
    parameters = {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "e.g., 'ehemann'"},
            "value": {"type": "string", "description": "e.g., 'Daniel'"}
        },
        "required": ["key", "value"]
    }

    async def execute(self, key=None, value=None, **kwargs) -> Any:
        if not key or not value or not self.cap.memory:
            return {"error": "Missing key/value or memory not available"}
        if len(str(value).split()) > 50:
            return {"error": "Value too long (>50 words)"}
        await self.cap.memory.learn_personal_data(key, str(value))
        return {"status": "success", "message": f"Saved '{key}'"}


class GetPersonalDataTool(McpTool):
    name = "get_personal_data"
    description = "Retrieve a stored personalized fact by key."
    parameters = {
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"]
    }

    async def execute(self, key=None, **kwargs) -> Any:
        if not key or not self.cap.memory:
            return {"error": "Missing key or memory not available"}
        val = await self.cap.memory.get_personal_data(key)
        return {"key": key, "value": val}


class GetSystemCapabilitiesTool(McpTool):
    name = "get_system_capabilities"
    description = "List all available system capabilities, supported intents, and domains. Use this to understand what this smart home assistant can do."
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> Any:
        # Dynamic introspection of available domains and features
        supported_domains = ["light", "cover", "switch", "climate", "vacuum", "timer", "media_player", "fan", "lock"]
        # In a real environment, we'd pull this from the integration's manifest or registry
        return {
            "intents": [
                "HassTurnOn", "HassTurnOff", "HassLightSet", "HassSetPosition", 
                "HassGetState", "HassClimateSetTemperature", "HassVacuumStart", 
                "HassTimerSet"
            ],
            "domains": supported_domains,
            "features": [
                "Multi-turn conversation", "Room/Floor awareness", 
                "Personal data memory (permanent storage of house facts)", 
                "Fuzzy name matching",
                "Calendar scheduling", 
                "Implicit intent resolution (e.g. 'mir ist kalt' -> 'Heizung wärmer')",
                "Area/Floor resolver",
                "Self-learning semantic cache (Muscle Memory)"
            ],
            "language": "German (primary), English (secondary)"
        }


class StoreCacheEntryTool(McpTool):
    name = "store_cache_entry"
    description = "Store a successful command resolution in the semantic cache. This teaches the fast-path (Stage 1) how to handle this phrasing in the future without cloud reasoning."
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The user's original phrasing"},
            "intent": {"type": "string", "description": "The resolved HA intent"},
            "entity_ids": {"type": "array", "items": {"type": "string"}, "description": "The resolved entity IDs"},
            "slots": {"type": "object", "description": "Extracted parameters (e.g., brightness, temperature)"},
            "domain": {"type": "string", "description": "The device domain"}
        },
        "required": ["text", "intent", "entity_ids"]
    }

    async def execute(self, text=None, intent=None, entity_ids=None, slots=None, domain=None, **kwargs) -> Any:
        if not self.cap.cache:
            return {"error": "Semantic cache not available"}
        
        # Store in cache
        await self.cap.cache.store(
            text=text,
            intent=intent,
            entity_ids=entity_ids,
            slots=slots or {},
            domain=domain
        )
        return {"status": "success", "message": f"Learned command: '{text}'"}


# --- Capability Class ---

class McpToolCapability(Capability):
    """Provides extensible MCP tools via a registry of tool handlers."""

    name = "mcp_tool"
    description = "Dynamic tool registry for LLM home discovery."

    # Registry of available tools
    TOOL_CLASSES: List[Type[McpTool]] = [
        ListAreasTool,
        ListEntitiesTool,
        GetEntityDetailsTool,
        ListAutomationsTool,
        GetAutomationDetailsTool,
        StorePersonalDataTool,
        GetPersonalDataTool,
        GetSystemCapabilitiesTool,
        StoreCacheEntryTool,
    ]

    def __init__(self, hass, config):
        super().__init__(hass, config)
        self.memory = None
        self.area_resolver = None
        self.cache = None
        # Instantiate tool handlers
        self.tools: Dict[str, McpTool] = {
            cls.name: cls(self) for cls in self.TOOL_CLASSES
        }

    def set_memory(self, memory_cap):
        self.memory = memory_cap

    def set_area_resolver(self, area_resolver):
        self.area_resolver = area_resolver

    def set_cache(self, cache_cap):
        self.cache = cache_cap

    def get_tools(self, domain: str = None, **kwargs) -> List[Dict[str, Any]]:
        """Return definitions for all registered tools."""
        return [t.get_definition() for t in self.tools.values()]

    async def execute_tool(self, name: str, args: Dict[str, Any]) -> Any:
        """Call a tool by name with arguments."""
        if name not in self.tools:
            return {"error": f"Tool '{name}' not found"}
        
        try:
            return await self.tools[name].execute(**args)
        except Exception as e:
            _LOGGER.exception("[McpTool] Error executing %s: %s", name, e)
            return {"error": str(e)}

    # --- LLM Reasoning Methods ---

    async def resolve_intent_via_llm(
        self,
        text: str,
        llm_config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Resolve user intent using MCP tools via multi-turn LLM reasoning."""
        from ..ollama_client import OllamaClient
        from ..utils.json_utils import extract_json_from_llm_string
        
        ip, port, model = llm_config.get("ip"), llm_config.get("port"), llm_config.get("model")
        if not ip or not port: return None
        client = OllamaClient(ip, int(port))
        
        tools_def = json.dumps(self.get_tools(), indent=2)
        system_prompt = f"""You are a professional Home Assistant smart home assistant. 
Determine the user's intent and extract required slots for controlling devices.

Available Intents: HassTurnOn, HassTurnOff, HassLightSet, HassSetPosition, HassGetState, HassClimateSetTemperature, HassVacuumStart, HassTimerSet.
Tools to explore the home: {tools_def}

OUTPUT: {{"tool": "name", "args": {{}}}} OR {{"intent": "Name", "slots": {{}}}}
"""
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Request: '{text}'"}]

        for _ in range(4):
            response = await client.chat_completion(model, messages, temperature=0.0)
            messages.append({"role": "assistant", "content": response})
            try:
                data = extract_json_from_llm_string(response)
                if "intent" in data: return data
                if "tool" in data:
                    res = await self.execute_tool(data["tool"], data.get("args", {}))
                    messages.append({"role": "user", "content": f"Result: {json.dumps(res)}"})
                    continue
            except: pass
        return None

    async def resolve_entity_via_llm(
        self, 
        text: str, 
        slots: Dict[str, Any], 
        intent: str, 
        domain: str,
        llm_config: Dict[str, Any]
    ) -> List[str]:
        """Resolve entity using MCP tools via multi-turn LLM loop."""
        from ..ollama_client import OllamaClient
        from ..utils.json_utils import extract_json_from_llm_string
        
        ip, port, model = llm_config.get("ip"), llm_config.get("port"), llm_config.get("model")
        if not ip or not port: return []
        client = OllamaClient(ip, int(port))
        
        tools_def = json.dumps(self.get_tools(domain=domain), indent=2)
        system_prompt = f"You are a Home Assistant expert. Find the correct entity ID for: '{text}' (Intent: {intent}). Tools to search the home: {tools_def}\nOUTPUT: {{\"tool\": \"...\"}} OR {{\"final_answer\": [\"id\"]}}"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Slots: {slots}"}]

        for _ in range(4):
            response = await client.chat_completion(model, messages, temperature=0.0)
            messages.append({"role": "assistant", "content": response})
            try:
                data = extract_json_from_llm_string(response)
                if "final_answer" in data: return data["final_answer"]
                if "tool" in data:
                    res = await self.execute_tool(data["tool"], data.get("args", {}))
                    messages.append({"role": "user", "content": f"Result: {json.dumps(res)}"})
                    continue
            except: pass
        return []

    async def run(self, user_input, **kwargs) -> Dict[str, Any]:
        return {}
