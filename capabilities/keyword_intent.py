import logging
from typing import Any, Dict, Optional, List

from .base import Capability
from .plural_detection import LIGHT_KEYWORDS, COVER_KEYWORDS, SENSOR_KEYWORDS, CLIMATE_KEYWORDS

_LOGGER = logging.getLogger(__name__)


class KeywordIntentCapability(Capability):
    """
    Detect a domain from German keywords and let the LLM pick
    a specific Home Assistant intent + slots within that domain.

    - If no clear domain is detected: return {} so Stage1 can escalate to Stage2.
    - If exactly one domain is detected: call the LLM with only that domain's intents.
    """

    name = "keyword_intent"
    description = "Derive a Home Assistant intent from a single German command using keyword domains."

    # Domain → list of keywords (singular + plural) reusing plural_detection helpers.
    DOMAIN_KEYWORDS: Dict[str, List[str]] = {
        "light": list(LIGHT_KEYWORDS.keys()) + list(LIGHT_KEYWORDS.values()),
        "cover": list(COVER_KEYWORDS.keys()) + list(COVER_KEYWORDS.values()),
        # Sensor keywords + specific adjectives/nouns indicating a measurement request
        "sensor": list(SENSOR_KEYWORDS.keys()) + list(SENSOR_KEYWORDS.values()) + ["grad", "warm", "kalt", "wieviel"],
        "climate": list(CLIMATE_KEYWORDS.keys()) + list(CLIMATE_KEYWORDS.values()) + ["klima"],
    }

    # Intents per domain + extra description + examples to tune the prompt.
    # Examples are rendered into the system prompt dynamically.
    INTENT_DOMAINS: Dict[str, Dict[str, Any]] = {
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
                "HassTurnOn": "Öffnet Rollläden (hochfahren).",
                "HassTurnOff": "Schließt Rollläden (runterfahren).",
                "HassSetPosition": "Setzt die Position eines Rollladens (0–100%).",
                "HassGetState": "Fragt den Zustand eines Rollladens ab.",
            },
            "examples": [
                'User: "Fahre den Rollladen im Wohnzimmer ganz runter"\n'
                '→ {"intent":"HassSetPosition","slots":{"area":"Wohnzimmer","position":0,"domain":"cover"}}',
                'User: "Öffne alle Jalousien im Schlafzimmer"\n'
                '→ {"intent":"HassTurnOn","slots":{"area":"Schlafzimmer","domain":"cover"}}',
            ],
        },
        "sensor": {
            "description": "Abfragen von Sensorwerten (Temperatur, Feuchtigkeit, Status).",
            "intents": {
                "HassGetState": "Fragt den Wert oder Zustand eines Sensors ab.",
            },
            "examples": [
                'User: "Wie ist die Temperatur im Büro?"\n'
                '→ {"intent":"HassGetState","slots":{"area":"Büro","domain":"sensor", "name":"Temperatur"}}',
                'User: "Wieviel Grad hat es im Wohnzimmer?"\n'
                '→ {"intent":"HassGetState","slots":{"area":"Wohnzimmer","domain":"sensor", "device_class":"temperature"}}',
            ],
        },
        "climate": {
            "description": "Steuerung von Heizkörpern, Thermostaten und Klimaanlagen.",
            "intents": {
                "HassClimateSetTemperature": "Setzt die Zieltemperatur.",
                "HassTurnOn": "Schaltet Heizung/Klima ein.",
                "HassTurnOff": "Schaltet Heizung/Klima aus.",
                "HassGetState": "Fragt Status der Heizung ab.",
            },
            "examples": [
                'User: "Stelle die Heizung im Bad auf 22 Grad"\n'
                '→ {"intent":"HassClimateSetTemperature","slots":{"area":"Bad","temperature":22}}',
                'User: "Mach die Heizung im Wohnzimmer aus"\n'
                '→ {"intent":"HassTurnOff","slots":{"area":"Wohnzimmer","domain":"climate"}}',
            ],
        },
    }

    # JSON-Schema for PromptExecutor/_safe_prompt
    SCHEMA: Dict[str, Any] = {
        "properties": {
            "intent": {"type": ["string", "null"]},
            "slots": {"type": "object"},
        },
    }

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _detect_domain(self, text: str) -> Optional[str]:
        """Keyword-based (with fuzzy) domain detection."""
        t = (text or "").lower()
        matches: List[str] = []

        # 1) Exact substring matching
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            if any(k in t for k in keywords):
                matches.append(domain)

        # Ambiguity handling:
        # If both 'climate' and 'sensor' match (e.g. "Heizung Temperatur"), prefer climate for control,
        # or sensor for queries? Usually explicit domain keywords win.
        # For now, strict: single match only, or return None to fall back.
        
        if len(matches) == 1:
            _LOGGER.debug(
                "[KeywordIntent] Detected domain '%s' from text=%r (exact match)",
                matches[0],
                text,
            )
            return matches[0]

        if matches:
            # If "sensor" and "climate" both match, prefer climate (control usually overrides query)
            if "climate" in matches and "sensor" in matches:
                 return "climate"
            
            _LOGGER.debug(
                "[KeywordIntent] Ambiguous exact domain match (%s) for text=%r → no domain chosen.",
                ", ".join(matches),
                text,
            )
            return None

        # 2) Fuzzy matching fallback
        import difflib
        import re

        tokens = re.findall(r"\w+", t)
        fuzzy_scores: Dict[str, float] = {}

        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            best_score = 0.0
            for kw in keywords:
                kw_l = kw.lower()
                for tok in tokens:
                    score = difflib.SequenceMatcher(None, kw_l, tok).ratio()
                    if score > best_score:
                        best_score = score
            if best_score >= 0.8:
                fuzzy_scores[domain] = best_score

        if len(fuzzy_scores) == 1:
            domain = next(iter(fuzzy_scores.keys()))
            score = fuzzy_scores[domain]
            _LOGGER.debug(
                "[KeywordIntent] Detected domain '%s' from text=%r via fuzzy match (score=%.2f)",
                domain,
                text,
                score,
            )
            return domain

        return None

    def _build_system_prompt(self, domain: str, meta: Dict[str, Any]) -> str:
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
            f"Current domain: {domain}",
        ]

        if desc:
            lines.append(f"Domain description: {desc}")

        lines.append("")
        lines.append("Allowed intents in this domain:")
        for iname, idesc in intents.items():
            lines.append(f"- {iname}: {idesc}")

        lines += [
            "",
            "You may only use these intents. Do NOT invent new intent names.",
            "",
            "Slots hints:",
            "- 'area': Raum- oder Bereichsname (z.B. 'Dusche', 'Wohnzimmer').",
            "- 'name': Gerätespezifischer Name, falls der Nutzer ein Gerät direkt benennt.",
            "- 'domain': HA-Domain wie 'light', 'cover', 'sensor', 'climate'.",
            "- 'device_class': (for sensors) e.g. 'temperature', 'humidity'.",
            "",
            "If you really cannot map the command to any allowed intent, respond with:",
            '{"intent": null, "slots": {}}',
        ]

        if examples:
            lines.append("")
            lines.append("Examples:")
            for ex in examples:
                lines.append(ex)

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def run(self, user_input, **_: Any) -> Dict[str, Any]:
        """Main entry: detect domain, then ask LLM for intent+slots within that domain."""
        text = user_input.text or ""
        domain = self._detect_domain(text)

        if not domain:
            return {}

        meta = self.INTENT_DOMAINS.get(domain)
        if not meta:
            _LOGGER.warning("[KeywordIntent] No INTENT_DOMAINS metadata for domain=%r", domain)
            return {}

        system = self._build_system_prompt(domain, meta)
        prompt = {
            "system": system,
            "schema": self.SCHEMA,
        }

        _LOGGER.debug("[KeywordIntent] Deriving intent for text=%r with domain=%s", text, domain)
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

        # Force domain injection if missing, to assist EntityResolver
        if "domain" not in slots:
             slots["domain"] = domain

        allowed = set((meta.get("intents") or {}).keys())
        if intent not in allowed:
            _LOGGER.warning(
                "[KeywordIntent] Model chose intent %r not in allowed set %s",
                intent,
                allowed,
            )
            return {}

        result = {
            "domain": domain,
            "intent": intent,
            "slots": slots,
        }
        _LOGGER.debug("[KeywordIntent] Final mapping: %s", result)
        return result
