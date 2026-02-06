"""Entity-scope phrase patterns for Semantic Cache.

Pattern Format: (Text Pattern, Intent, Slots)
Placeholders:
- {device}: Accusative device word (e.g. "das Licht", "den Rollladen")
- {device_nom}: Nominative device word (e.g. "der Rollladen")
- {entity_name}: Entity friendly name.
- {area}: Area name (Optional in this context, but used for specific "Entity in Area" phrasing)

NOTE: We generate two variants for each pattern:
1. With Area: "Schalte die Deckenlampe im Büro an"
2. Global/Unique: "Schalte die Deckenlampe an"
"""

ENTITY_PHRASE_PATTERNS = {
    "light": [
        # === ON/OFF ===
        ("Schalte {device} {entity_name} in {area} an", "HassTurnOn", {}),
        ("Schalte {device} {entity_name} in {area} aus", "HassTurnOff", {}),
        ("Schalte {entity_name} an", "HassTurnOn", {}),
        ("Schalte {entity_name} aus", "HassTurnOff", {}),
        
        # === ABSOLUTE CONTROL (Brightness) ===
        ("Stelle {device} {entity_name} in {area} auf 50 Prozent", "HassLightSet", {"brightness": 50}),
        ("Stelle {entity_name} auf 50 Prozent", "HassLightSet", {"brightness": 50}),
        
        # === RELATIVE CONTROL (Brightness) ===
        ("Mach {device} {entity_name} in {area} heller", "HassLightSet", {"command": "step_up"}),
        ("Mach {device} {entity_name} in {area} dunkler", "HassLightSet", {"command": "step_down"}),
        ("Mach {entity_name} heller", "HassLightSet", {"command": "step_up"}),
        
        # === TIME-BASED ===
        ("Schalte {device} {entity_name} in {area} in 10 Minuten an", "DelayedControl", {"command": "on"}),
        ("Schalte {device} {entity_name} in 10 Minuten aus", "DelayedControl", {"command": "off"}),
        ("Schalte {entity_name} für 10 Minuten an", "TemporaryControl", {"command": "on"}),
        
        # === QUERY ===
        ("Ist {device_nom} {entity_name} an?", "HassGetState", {"state": "on"}),
        ("Ist {entity_name} an?", "HassGetState", {"state": "on"}),
    ],
    
    "cover": [
        # === OPEN/CLOSE ===
        ("Öffne {device} {entity_name} in {area}", "HassTurnOn", {}),
        ("Schließe {device} {entity_name} in {area}", "HassTurnOff", {}),
        ("Öffne {entity_name}", "HassTurnOn", {}),
        ("Schließe {entity_name}", "HassTurnOff", {}),
        
        # === ABSOLUTE CONTROL (Position) ===
        ("Stelle {device} {entity_name} in {area} auf 50 Prozent", "HassSetPosition", {"position": 50}),
        ("Stelle {entity_name} auf 50 Prozent", "HassSetPosition", {"position": 50}),
        
        # === RELATIVE CONTROL (Position) ===
        ("Fahre {device} {entity_name} weiter hoch", "HassSetPosition", {"command": "step_up"}),
        ("Fahre {entity_name} weiter runter", "HassSetPosition", {"command": "step_down"}),
        
        # === TIME-BASED ===
        ("Öffne {entity_name} in 10 Minuten", "DelayedControl", {"command": "on"}),
        ("Schließe {entity_name} in 10 Minuten", "DelayedControl", {"command": "off"}),
        ("Öffne {entity_name} für 10 Minuten", "TemporaryControl", {"command": "on"}),
        
        # === QUERY ===
        ("Ist {device_nom} {entity_name} offen?", "HassGetState", {"state": "open"}),
    ],
    
    "climate": [
        # === ON/OFF ===
        ("Schalte {device} {entity_name} in {area} an", "HassTurnOn", {}),
        ("Schalte {device} {entity_name} aus", "HassTurnOff", {}),
        
        # === ABSOLUTE (Temperature) ===
        ("Stelle {device} {entity_name} auf 21 Grad", "HassClimateSetTemperature", {}),
        
        # === RELATIVE (Temperature) ===
        ("Mach {device} {entity_name} wärmer", "HassClimateSetTemperature", {"command": "step_up"}),
        
        # === TIME-BASED ===
        ("Schalte {entity_name} in 10 Minuten an", "DelayedControl", {"command": "on"}),
        
        # === QUERY ===
        ("Ist {entity_name} an?", "HassGetState", {"state": "on"}),
    ],
    
    "fan": [
        ("Schalte {device} {entity_name} an", "HassTurnOn", {}),
        ("Schalte {device} {entity_name} aus", "HassTurnOff", {}),
        ("Mach {entity_name} schneller", "HassFanSetPercentage", {"command": "step_up"}),
        ("Schalte {entity_name} in 10 Minuten aus", "DelayedControl", {"command": "off"}),
    ],
    
    "media_player": [
        ("Schalte {device} {entity_name} an", "HassTurnOn", {}),
        ("Pausiere {entity_name}", "HassMediaPause", {}),
        ("Mach {entity_name} lauter", "HassMediaSetVolume", {"command": "volume_up"}),
        ("Schalte {entity_name} in 30 Minuten aus", "DelayedControl", {"command": "off"}),
    ],
    
    "switch": [
        ("Schalte {device} {entity_name} an", "HassTurnOn", {}),
        ("Schalte {entity_name} aus", "HassTurnOff", {}),
        ("Schalte {entity_name} in 10 Minuten aus", "DelayedControl", {"command": "off"}),
        ("Schalte {entity_name} für 15 Minuten an", "TemporaryControl", {"command": "on"}),
    ],
    
    "automation": [
        ("Aktiviere {device} {entity_name}", "HassTurnOn", {}),
        ("Deaktiviere {entity_name}", "HassTurnOff", {}),
        ("Aktiviere {entity_name} in 10 Minuten", "DelayedControl", {"command": "on"}),
    ],
}
