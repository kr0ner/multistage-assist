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

    def get_tools(self) -> List[Dict[str, Any]]:
        """Return tool definitions (compatible with Gemini/OpenAI)."""
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
        tools_def = json.dumps(self.get_tools(), indent=2)
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
        return {"error": f"Unknown tool: {tool_name}"}
