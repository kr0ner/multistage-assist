import logging
from typing import Any, Dict, Optional, List

from .base import Capability
from .plural_detection import LIGHT_KEYWORDS, COVER_KEYWORDS

_LOGGER = logging.getLogger(__name__)


class KeywordIntentCapability(Capability):
    """
    Detect a domain/group from German keywords and let the LLM pick
    a specific Home Assistant intent + slots within that group.

    - If no clear group is detected: return {} so Stage1 can escalate to Stage2.
    - If exactly one group is detected: call the LLM with only that group's intents.
    """

    name = "keyword_intent"
    description = "Derive a Home Assistant intent from a single German command using keyword groups."

    # Group → list of keywords (singular + plural) reusing plural_detection helpers.
    # LIGHT_KEYWORDS and COVER_KEYWORDS are expected to be dicts:
    #   singular -> plural
    GROUP_KEYWORDS: Dict[str, List[str]] = {
        "light": list(LIGHT_KEYWORDS.keys()) + list(LIGHT_KEYWORDS.values()),
        "cover": list(COVER_KEYWORDS.keys()) + list(COVER_KEYWORDS.values()),
    }

    # Intents per group + extra description + examples to tune the prompt.
    # Examples are rendered into the system prompt dynamically.
    INTENT_GROUPS: Dict[str, Dict[str, Any]] = {
        "light": {
            "description": "Steuerung von Lichtern, Lampen und Beleuchtung.",
            "intents": {
                "HassTurnOn": "Schaltet eine oder mehrere Lichter/Lampen ein.",
                "HassTurnOff": "Schaltet eine oder mehrere Lichter/Lampen aus.",
                "HassLightSet": "Setzt Helligkeit oder Farbe eines Lichts.",
                "HassGetState": "Fragt den Zustand eines Lichts ab.",
            },
            "examples": [
                'User: "Schalte das Licht in der Dusche an"\n'
                '→ {"intent":"HassTurnOn","slots":{"name":"Dusche","domain":"light"}}',
                'User: "Wie ist der Status vom Licht im Wohnzimmer?"\n'
                '→ {"intent":"HassGetState","slots":{"area":"Wohnzimmer","domain":"light"}}',
            ],
        },
        "cover": {
            "description": "Steuerung von Rollläden, Rollos und Jalousien.",
            "intents": {
                "HassTurnOn": "Öffnet Rollläden, Rollos oder Jalousien.",
                "HassTurnOff": "Schließt Rollläden, Rollos oder Jalousien.",
                "HassSetPosition": "Setzt die Position eines Rollladens (0–100%).",
                "HassGetState": "Fragt den Zustand eines Rollladens ab.",
            },
            "examples": [
                'User: "Fahre den Rollladen im Wohnzimmer ganz runter"\n'
                '→ {"intent":"HassSetPosition","slots":{"area":"Wohnzimmer","position":0}}',
                'User: "Öffne alle Jalousien im Schlafzimmer"\n'
                '→ {"intent":"HassTurnOn","slots":{"area":"Schlafzimmer","domain":"cover"}}',
            ],
        },
    }

    # JSON-Schema for PromptExecutor/_safe_prompt
    # - "intent" is string or null; null means "no mapping possible".
    SCHEMA: Dict[str, Any] = {
        "properties": {
            "intent": {"type": ["string", "null"]},
            "slots": {"type": "object"},
        },
    }

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _detect_group(self, text: str) -> Optional[str]:
        """Very simple keyword-based group detection.

        If we don't get exactly ONE match, we return None to avoid ambiguity.
        """
        t = (text or "").lower()
        matches: List[str] = []

        for group, keywords in self.GROUP_KEYWORDS.items():
            if any(k in t for k in keywords):
                matches.append(group)

        if len(matches) == 1:
            _LOGGER.debug("[KeywordIntent] Detected group '%s' from text=%r", matches[0], text)
            return matches[0]

        if not matches:
            _LOGGER.debug("[KeywordIntent] No group keyword match for text=%r", text)
        else:
            _LOGGER.debug(
                "[KeywordIntent] Ambiguous group match (%s) for text=%r → no group chosen.",
                ", ".join(matches),
                text,
            )
        return None

    def _build_system_prompt(self, group: str, meta: Dict[str, Any]) -> str:
        """Dynamically build the system prompt from Python metadata."""
        desc = meta.get("description") or ""
        intents: Dict[str, str] = meta.get("intents") or {}
        examples: List[str] = meta.get("examples") or []

        lines: List[str] = [
            "You are a language model that selects a Home Assistant intent",
            "for a single German smart home voice command.",
            "",
            "The user speaks German (e.g. \"Schalte das Licht in der Dusche an\").",
            "You MUST:",
            "1. Understand what the user wants to do.",
            "2. Choose EXACTLY ONE intent from the allowed list.",
            "3. Fill reasonable slots for Home Assistant (area, name, domain, etc.).",
            "",
            f"Current group: {group}",
        ]

        if desc:
            lines.append(f"Group description: {desc}")

        lines.append("")
        lines.append("Allowed intents in this group:")
        for iname, idesc in intents.items():
            lines.append(f"- {iname}: {idesc}")

        lines += [
            "",
            "You may only use these intents. Do NOT invent new intent names.",
            "",
            "Slots hints:",
            "- 'area': Raum- oder Bereichsname (z.B. 'Dusche', 'Wohnzimmer').",
            "- 'name': Gerätespezifischer Name, falls der Nutzer ein Gerät direkt benennt.",
            "- 'domain': HA-Domain wie 'light', 'cover', usw.",
            "- Weitere Slots nur, wenn sie sinnvoll sind (z.B. 'position' für Rollläden, 'brightness' für Licht).",
            "",
            "If you really cannot map the command to any allowed intent, respond with:",
            '{"intent": null, "slots": {}}',
        ]

        if examples:
            lines.append("")
            lines.append("Examples:")
            for ex in examples:
                lines.append(ex)

        # No explicit "output format" here – PromptExecutor will add that
        # automatically based on SCHEMA.
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def run(self, user_input, **_: Any) -> Dict[str, Any]:
        """Main entry: detect group, then ask LLM for intent+slots within that group."""
        text = user_input.text or ""
        group = self._detect_group(text)

        # No clear group → let Stage1 escalate to Stage2 or other mechanisms.
        if not group:
            return {}

        meta = self.INTENT_GROUPS.get(group)
        if not meta:
            _LOGGER.warning("[KeywordIntent] No INTENT_GROUPS metadata for group=%r", group)
            return {}

        system = self._build_system_prompt(group, meta)
        prompt = {
            "system": system,
            "schema": self.SCHEMA,
        }

        _LOGGER.debug("[KeywordIntent] Deriving intent for text=%r with group=%s", text, group)
        data = await self._safe_prompt(prompt, {"user_input": text})

        if not isinstance(data, dict):
            _LOGGER.warning("[KeywordIntent] Model did not return a dict: %r", data)
            return {}

        intent = data.get("intent")
        if intent is None:
            _LOGGER.debug("[KeywordIntent] Model returned null intent for text=%r", text)
            return {}

        slots = data.get("slots") or {}
        if not isinstance(slots, dict):
            _LOGGER.warning("[KeywordIntent] Invalid slots type from model: %r", slots)
            slots = {}

        allowed = set((meta.get("intents") or {}).keys())
        if intent not in allowed:
            _LOGGER.warning(
                "[KeywordIntent] Model chose intent %r not in allowed set %s",
                intent,
                allowed,
            )
            return {}

        result = {
            "group": group,
            "intent": intent,
            "slots": slots,
        }
        _LOGGER.debug("[KeywordIntent] Final mapping: %s", result)
        return result
