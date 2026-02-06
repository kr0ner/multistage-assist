import logging
from typing import Any, Dict, List, Optional, Set
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
    floor_registry as fr,
)
from homeassistant.core import HomeAssistant

from .base import Capability

# Import exposure check
try:
    from homeassistant.components.homeassistant.exposed_entities import async_should_expose
except ImportError:
    def async_should_expose(hass, conversation_agent_id, entity_id):
        """Fallback: always return True."""
        return True

_LOGGER = logging.getLogger(__name__)


class McpToolCapability(Capability):
    """Provides MCP tools for entity and area lookup."""

    name = "mcp_tool"
    description = "Lookup tools for entities and areas."

    async def run(self, user_input, **kwargs):
        """Not used directly. Methods are called individually."""
        return {}

    async def list_areas(self) -> List[Dict[str, Any]]:
        """List all areas with their IDs and attached floors."""
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

    async def list_entities(
        self, 
        domain: Optional[str] = None, 
        area_name: Optional[str] = None,
        device_class: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """List entities filtered by domain, area name, or device class.
        
        Args:
            domain: Filter by domain (e.g. 'light')
            area_name: Filter by area name (fuzzy matched)
            device_class: Filter by device class
            limit: Max results to return
        """
        er_reg = er.async_get(self.hass)
        dr_reg = dr.async_get(self.hass)
        ar_reg = ar.async_get(self.hass)
        
        # 1. Resolve Area Name to ID(s) if provided
        target_area_ids = set()
        if area_name:
            norm_area_name = area_name.lower().strip()
            for area in ar_reg.areas.values():
                if (norm_area_name in area.name.lower() or 
                    any(norm_area_name in alias.lower() for alias in area.aliases)):
                    target_area_ids.add(area.id)
            
            if not target_area_ids:
                return []  # Area specified but not found

        results = []
        
        for entry in er_reg.entities.values():
            if entry.disabled_by:
                continue
            
            # Check exposure
            if not async_should_expose(self.hass, "conversation", entry.entity_id):
                continue

                
            # Filter by Domain
            if domain and entry.domain != domain:
                continue
            
            # Filter by Area
            if target_area_ids:
                match = False
                # Direct area match
                if entry.area_id in target_area_ids:
                    match = True
                # Device area match
                elif entry.device_id:
                    device = dr_reg.async_get(entry.device_id)
                    if device and device.area_id in target_area_ids:
                        match = True
                
                if not match:
                    continue

            # Filter by Device Class (requires state lookup)
            state = self.hass.states.get(entry.entity_id)
            if device_class:
                if not state:
                    continue
                dc = state.attributes.get("device_class")
                if not dc or dc != device_class:
                    continue

            # Add to results
            name = entry.original_name or entry.name or entry.entity_id
            if state and state.attributes.get("friendly_name"):
                name = state.attributes.get("friendly_name")

            results.append({
                "entity_id": entry.entity_id,
                "name": name,
                "area_id": entry.area_id,
                "domain": entry.domain,
                "aliases": list(entry.aliases) if entry.aliases else [],
            })
            
            if len(results) >= limit:
                break
                
        return results

    async def list_automations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List all available automations."""
        er_reg = er.async_get(self.hass)
        results = []
        
        # Get all automation entities
        automations = [e for e in er_reg.entities.values() if e.domain == "automation"]
        
        # Iterate and verify exposure
        for entry in automations:
            # Check exposure relative to "conversation" agent
            if not async_should_expose(self.hass, "conversation", entry.entity_id):
                continue
                
            state = self.hass.states.get(entry.entity_id)
            name = entry.original_name or entry.name or entry.entity_id
            if state and state.attributes.get("friendly_name"):
                name = state.attributes.get("friendly_name")
            
            description = state.attributes.get("description", "") if state else ""
            
            results.append({
                "entity_id": entry.entity_id,
                "name": name,
                "description": description,
                "state": state.state if state else "unknown"
            })
            
            if len(results) >= limit:
                break
                
        return results

    async def get_automation_details(self, entity_id: str) -> Dict[str, Any]:
        """Get detailed configuration of an automation."""
        state = self.hass.states.get(entity_id)
        if not state:
            return {"error": "Automation not found"}
        
        # Verify exposure
        if not async_should_expose(self.hass, "conversation", entity_id):
             return {"error": "Automation not exposed to voice"}

        er_reg = er.async_get(self.hass)
        entry = er_reg.async_get(entity_id)
        
        # Try to get existing config if possible (limited access from HA core)
        # We return attributes which might contain useful info
        details = {
            "entity_id": entity_id,
            "name": state.attributes.get("friendly_name", entity_id),
            "state": state.state,
            "last_triggered": str(state.attributes.get("last_triggered", "never")),
            "mode": state.attributes.get("mode", "single"),
            "attributes": dict(state.attributes)
        }
        return details

    async def get_entity_details(self, entity_id: str) -> Dict[str, Any]:
        """Get detailed info about a specific entity."""
        state = self.hass.states.get(entity_id)
        if not state:
            return {"error": "Entity not found"}
            
        er_reg = er.async_get(self.hass)
        entry = er_reg.async_get(entity_id)
        
        details = {
            "entity_id": entity_id,
            "state": state.state,
            "attributes": dict(state.attributes),
            "last_changed": state.last_changed.isoformat(),
        }
        
        if entry:
            details.update({
                "name_registry": entry.name,
                "original_name": entry.original_name,
                "platform": entry.platform,
                "device_id": entry.device_id,
                "area_id": entry.area_id,
                "aliases": list(entry.aliases) if entry.aliases else [],
            })
            
        return details

    def get_tools(self, intent: str = None, domain: str = None) -> List[Dict[str, Any]]:
        """Return tool definitions (compatible with Gemini/OpenAI).
        
        Args:
            intent: Optional intent context to filter tools
            domain: Optional domain context to filter tools
        """
        tools = []
        
        # 1. Area Tools (Always useful)
        tools.append({
            "name": "list_areas",
            "description": "List all areas (rooms) in the smart home with their floors.",
            "parameters": {"type": "object", "properties": {}}
        })
        
        # 2. Automation Tools (Only if domain is automation or unknown)
        if not domain or domain == "automation":
            tools.extend([
                {
                    "name": "list_automations",
                    "description": "List all automations exposed to voice assistants.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {"type": "integer", "description": "Max results (default 50)"}
                        }
                    }
                },
                {
                    "name": "get_automation_details",
                    "description": "Get details/config of a specific automation.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entity_id": {"type": "string", "description": "The automation entity ID"}
                        },
                        "required": ["entity_id"]
                    }
                }
            ])
            
        # 3. Entity Tools (Only if domain is NOT automation)
        if domain != "automation":
            tools.extend([
                {
                    "name": "list_entities",
                    "description": "List entities filtered by domain, area name, or device class.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "domain": {"type": "string", "description": "Filter by domain (light, cover, switch, climate, sensor)"},
                            "area_name": {"type": "string", "description": "Filter by area/room name"},
                            "device_class": {"type": "string", "description": "Filter by device class (temperature, humidity, power)"},
                            "limit": {"type": "integer", "description": "Max results (default 50)"}
                        }
                    }
                },
                {
                    "name": "get_entity_details",
                    "description": "Get detailed info about a specific entity including its current state.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entity_id": {"type": "string", "description": "The entity ID to look up"}
                        },
                        "required": ["entity_id"]
                    }
                }
            ])
            
        _LOGGER.debug("[McpToolCapability] Returning %d tools (domain=%s, intent=%s)", len(tools), domain, intent)
        return tools

    async def resolve_intent_via_llm(
        self,
        text: str,
        llm_config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Resolve user intent using MCP tools via multi-turn LLM reasoning.
        
        This is used when keyword_intent fails to parse intent from the utterance.
        The LLM can use tools to explore the smart home state and derive the intent.
        
        Args:
            text: User input text
            llm_config: Dict containing 'ip', 'port', 'model' for Ollama
            
        Returns:
            Dict with 'intent', 'slots', 'domain' if successful, None otherwise
        """
        from ..ollama_client import OllamaClient
        import json
        
        ip = llm_config.get("ip")
        port = llm_config.get("port")
        model = llm_config.get("model")
        
        if not ip or not port:
            _LOGGER.warning("[McpToolCapability] Missing Ollama config, skipping intent reasoning.")
            return None
            
        client = OllamaClient(ip, int(port))
        
        # System prompt for intent reasoning
        # Pass unknown domain/intent so it gets useful generalized tools
        tools_def = json.dumps(self.get_tools(intent=None, domain=None), indent=2)
        system_prompt = f"""You are a smart home assistant. Analyze the user's request and determine the intent.

Available Intents (ONLY use these):
- HassTurnOn: Turn on (lights, switches, etc.)
- HassTurnOff: Turn off (lights, switches, etc.)
- HassLightSet: Set brightness or color
- HassSetPosition: Set position (covers, blinds)
- HassGetState: Query state (temperature, status)
- HassClimateSetTemperature: Set thermostat temperature
- HassVacuumStart: Start vacuum
- HassVacuumStop: Stop vacuum

Available Slots:
- area: Room/area name (Küche, Bad, Büro, etc.)
- name: Specific device name (optional)
- domain: Device type (light, cover, climate, switch, sensor, vacuum)
- device_class: For sensors (temperature, humidity, power)
- brightness: 0-100 for lights
- temperature: Degrees for climate
- position: 0-100 for covers

Tools you can use:
{tools_def}

Instructions:
1. Analyze the user's natural language request.
2. If needed, use tools to understand the smart home context (e.g., what areas exist, what devices are available).
3. Determine the most likely intent and slots.

OUTPUT FORMAT (Strict JSON):
- To Call Tool: {{"tool": "tool_name", "args": {{...}}}}
- To Finish: {{"intent": "IntentName", "slots": {{"area": "...", "domain": "..."}}, "reasoning": "brief explanation"}}
- If unsure or cannot determine: {{"intent": null, "reason": "explanation"}}

Do not output any text outside the JSON.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User request: '{text}'"}
        ]

        # ReAct loop
        max_turns = 4
        for i in range(max_turns):
            _LOGGER.debug("[McpToolCapability] Intent reasoning Turn %d/%d", i+1, max_turns)
            
            try:
                response = await client.chat_completion(model, messages, temperature=0.0)
            except Exception as e:
                _LOGGER.error("[McpToolCapability] Ollama chat failed: %s", e)
                return None
                
            messages.append({"role": "assistant", "content": response})
            _LOGGER.debug("[McpToolCapability] LLM Response: %s", response[:200] if len(response) > 200 else response)
            
            # Parse JSON
            try:
                cleaned = response.strip()
                if "{" in cleaned and "}" in cleaned:
                    cleaned = cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]
                
                data = json.loads(cleaned)
            except (json.JSONDecodeError, ValueError):
                _LOGGER.warning("[McpToolCapability] Invalid JSON from LLM in intent reasoning")
                messages.append({"role": "user", "content": "Error: Invalid JSON. Please output ONLY valid JSON."})
                continue
            
            # Check for final answer
            if "intent" in data:
                intent = data.get("intent")
                if intent:
                    _LOGGER.info("[McpToolCapability] Intent reasoning resolved: %s", intent)
                    return {
                        "intent": intent,
                        "slots": data.get("slots", {}),
                        "domain": data.get("slots", {}).get("domain"),
                        "reasoning": data.get("reasoning", "")
                    }
                else:
                    _LOGGER.info("[McpToolCapability] Intent reasoning returned null intent: %s", data.get("reason", "unknown"))
                    return None
                
            # Handle tool calls
            if "tool" in data:
                tool_name = data["tool"]
                args = data.get("args", {})
                _LOGGER.info("[McpToolCapability] Intent reasoning calling tool '%s' with args %s", tool_name, args)
                
                try:
                    tool_result = await self._execute_internal_tool(tool_name, args)
                    result_str = json.dumps(tool_result, ensure_ascii=False)
                    # Truncate if too long
                    if len(result_str) > 2000:
                        result_str = result_str[:2000] + "... (truncated)"
                        
                    messages.append({"role": "user", "content": f"Tool '{tool_name}' Output: {result_str}"})
                except Exception as e:
                    messages.append({"role": "user", "content": f"Tool execution error: {str(e)}"})
            else:
                messages.append({"role": "user", "content": "Error: Output must contain 'tool' or 'intent'."})

        _LOGGER.warning("[McpToolCapability] Intent reasoning exhausted turns without result.")
        return None

    async def resolve_entity_via_llm(
        self, 
        text: str, 
        slots: Dict[str, Any], 
        intent: str, 
        domain: str,
        llm_config: Dict[str, Any]
    ) -> List[str]:
        """Resolve entity using MCP tools via multi-turn LLM loop.
        
        Args:
            text: User input text
            slots: Extracted slots
            intent: Detected intent
            domain: Detected domain
            llm_config: Dict containing 'ip', 'port', 'model' for Ollama
        """
        # Lazy import to avoid circular dependency if any
        from ..ollama_client import OllamaClient
        import json
        
        ip = llm_config.get("ip")
        port = llm_config.get("port")
        model = llm_config.get("model")
        
        if not ip or not port:
            _LOGGER.warning("[McpToolCapability] Missing Ollama config, skipping recovery.")
            return []
            
        client = OllamaClient(ip, int(port))
        
        # 2. System Prompt
        # Filter tools to only those relevant for this domain
        tools_def = json.dumps(self.get_tools(intent=intent, domain=domain), indent=2)
        system_prompt = f"""You are a smart home agent assistant.
Your goal is to find the correct entity ID for the user's request using the available tools.

Tools:
{tools_def}

Instructions:
1. Analyze the User Request, Intent, and Domain.
2. Decide which tool to call to find the entity (e.g., list_entities).
3. OUTPUT FORMAT (Strict JSON):
   - To Call Tool: {{"tool": "tool_name", "args": {{...}}}}
   - To Finish: {{"final_answer": ["entity_id", ...]}}

4. If you cannot find any matching entity after using tools, return empty list: {{"final_answer": []}}

Do not output any text outside the JSON.
"""

        # 3. Message History
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User Request: '{text}'\nIntent: {intent}\nDomain: {domain}\nSlots: {slots}\n\nResolve the entity ID."}
        ]

        # 4. ReAct Loop
        max_turns = 5
        for i in range(max_turns):
            _LOGGER.debug("[McpToolCapability] ReAct Turn %d/5", i+1)
            
            try:
                response = await client.chat_completion(model, messages, temperature=0.0)
            except Exception as e:
                _LOGGER.error("[McpToolCapability] Ollama chat failed: %s", e)
                break
                
            messages.append({"role": "assistant", "content": response})
            _LOGGER.debug("[McpToolCapability] LLM Response: %s", response)
            
            # Parse JSON
            try:
                cleaned = response.strip()
                if "{" in cleaned and "}" in cleaned:
                    cleaned = cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]
                
                data = json.loads(cleaned)
            except (json.JSONDecodeError, ValueError):
                _LOGGER.warning("[McpToolCapability] Invalid JSON from LLM")
                messages.append({"role": "user", "content": "Error: Invalid JSON. Please output ONLY valid JSON."})
                continue
            
            if "final_answer" in data:
                ids = data["final_answer"]
                _LOGGER.info("[McpToolCapability] ReAct loop finished. Result: %s", ids)
                return ids
                
            if "tool" in data:
                tool_name = data["tool"]
                args = data.get("args", {})
                _LOGGER.info("[McpToolCapability] Calling tool '%s' with args %s", tool_name, args)
                
                try:
                    tool_result = await self._execute_internal_tool(tool_name, args)
                    result_str = json.dumps(tool_result, ensure_ascii=False)
                    # Truncate if too long
                    if len(result_str) > 2000:
                        result_str = result_str[:2000] + "... (truncated)"
                        
                    messages.append({"role": "user", "content": f"Tool '{tool_name}' Output: {result_str}"})
                except Exception as e:
                    messages.append({"role": "user", "content": f"Tool execution error: {str(e)}"})
            else:
                 messages.append({"role": "user", "content": "Error: Output must contain 'tool' or 'final_answer'."})

        _LOGGER.warning("[McpToolCapability] ReAct loop exhausted turns without final answer.")
        return []

    async def _execute_internal_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        if tool_name == "list_areas":
            return await self.list_areas()
        elif tool_name == "list_entities":
            # Sanitize args
            valid_args = ["domain", "area_name", "device_class", "limit"]
            clean_args = {k: v for k, v in args.items() if k in valid_args}
            return await self.list_entities(**clean_args)
        elif tool_name == "get_entity_details":
            if "entity_id" not in args: return {"error": "Missing entity_id"}
            return await self.get_entity_details(args["entity_id"])
        elif tool_name == "list_automations":
            limit = args.get("limit", 50)
            return await self.list_automations(limit=limit)
        elif tool_name == "get_automation_details":
             if "entity_id" not in args: return {"error": "Missing entity_id"}
             return await self.get_automation_details(args["entity_id"])
        return {"error": f"Unknown tool: {tool_name}"}
