"""Global-scope phrase patterns for Semantic Cache.

Pattern Format: (Text Pattern, Intent, Slots)
"""

GLOBAL_PHRASE_PATTERNS = {
    "light": [
        ("Schalte alle Lichter aus", "HassTurnOff", {}),
        ("Schalte alle Lichter an", "HassTurnOn", {}),
        ("Mach alle Lichter heller", "HassLightSet", {"command": "step_up"}),
        ("Mach alle Lichter dunkler", "HassLightSet", {"command": "step_down"}),
        ("Dimme alle Lichter auf 50 Prozent", "HassLightSet", {"brightness": 50}),
        ("Stelle alle Lichter auf 50 Prozent", "HassLightSet", {"brightness": 50}),
        # State queries
        ("Welche Lichter sind an?", "HassGetState", {"state": "on"}),
        ("Sind alle Lichter an?", "HassGetState", {"state": "on"}),
    ],
    "cover": [
        ("Schließe alle Rollläden", "HassTurnOff", {}),
        ("Öffne alle Rollläden", "HassTurnOn", {}),
        ("Fahre alle Rollläden weiter hoch", "HassSetPosition", {"command": "step_up"}),
        ("Fahre alle Rollläden weiter runter", "HassSetPosition", {"command": "step_down"}),
        ("Stelle alle Rollläden auf 50 Prozent", "HassSetPosition", {"position": 50}),
        ("Welche Rollläden sind offen?", "HassGetState", {"state": "open"}),
    ],
    "switch": [
        ("Schalte alle Schalter aus", "HassTurnOff", {}),
        ("Schalte alle Schalter an", "HassTurnOn", {}),
    ],
    "fan": [
        ("Schalte alle Ventilatoren aus", "HassTurnOff", {}),
        ("Schalte alle Ventilatoren an", "HassTurnOn", {}),
    ],
    "media_player": [
        ("Schalte alle Fernseher aus", "HassTurnOff", {}),
        ("Schalte alle Fernseher an", "HassTurnOn", {}),
        ("Schalte alle Geräte aus", "HassTurnOff", {}),
        ("Schalte alle Geräte an", "HassTurnOn", {}),
    ],
    "automation": [
        ("Deaktiviere alle Automatisierungen", "HassTurnOff", {}),
        ("Aktiviere alle Automatisierungen", "HassTurnOn", {}),
    ],
}
