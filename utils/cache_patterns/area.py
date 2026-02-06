"""Area-scope phrase patterns for Semantic Cache.

Pattern Format: (Text Pattern, Intent, Slots)
Placeholders:
- {device}: Accusative device word (e.g. "die Lichter", "den Rollladen")
- {device_nom}: Nominative device word (e.g. "der Rollladen")
- {area}: Area name
"""

AREA_PHRASE_PATTERNS = {
    "light": [
        # === ON/OFF ===
        ("Schalte {device} in {area} an", "HassTurnOn", {}),
        ("Schalte {device} in {area} aus", "HassTurnOff", {}),
        ("Mach {device} in {area} an", "HassTurnOn", {}),
        ("Mach {device} in {area} aus", "HassTurnOff", {}),
        
        # === POLITE / INDIRECT (Explicitly map to Command) ===
        ("Kannst du {device} in {area} anschalten?", "HassTurnOn", {}),
        ("Kannst du {device} in {area} anmachen?", "HassTurnOn", {}),
        ("Würdest du bitte {device} in {area} anmachen?", "HassTurnOn", {}),
        ("Kannst du {device} in {area} ausschalten?", "HassTurnOff", {}),
        ("Kannst du {device} in {area} ausmachen?", "HassTurnOff", {}),
        
        # === ABSOLUTE CONTROL (Brightness) ===
        ("Stelle {device} in {area} auf 50 Prozent", "HassLightSet", {"brightness": 50}),
        ("Setze {device} in {area} auf 100 Prozent", "HassLightSet", {"brightness": 100}),
        
        # === RELATIVE CONTROL (Brightness) ===
        ("Mach {device} in {area} heller", "HassLightSet", {"command": "step_up"}),
        ("Mach {device} in {area} dunkler", "HassLightSet", {"command": "step_down"}),
        ("Dimme {device} in {area} hoch", "HassLightSet", {"command": "step_up"}),
        ("Dimme {device} in {area} runter", "HassLightSet", {"command": "step_down"}),
        
        # === TIME-BASED ===
        ("Schalte {device} in {area} in 10 Minuten an", "DelayedControl", {"command": "on"}),
        ("Schalte {device} in {area} in 10 Minuten aus", "DelayedControl", {"command": "off"}),
        ("Schalte {device} in {area} für 10 Minuten an", "TemporaryControl", {"command": "on"}),
        ("Schalte {device} in {area} für 10 Minuten aus", "TemporaryControl", {"command": "off"}),
        
        # === QUERY ===
        ("Ist {device_nom} in {area} an?", "HassGetState", {"state": "on"}),
        ("Sind {device} in {area} an?", "HassGetState", {"state": "on"}),
    ],
    
    "cover": [
        # === OPEN/CLOSE (Using HassTurnOn/Off as requested) ===
        # TurnOn = Open, TurnOff = Close
        ("Öffne {device} in {area}", "HassTurnOn", {}),
        ("Mach {device} in {area} auf", "HassTurnOn", {}),
        ("Fahre {device} in {area} hoch", "HassTurnOn", {}),
        
        ("Schließe {device} in {area}", "HassTurnOff", {}),
        ("Mach {device} in {area} zu", "HassTurnOff", {}),
        ("Fahre {device} in {area} runter", "HassTurnOff", {}),
        
        # === POLITE / INDIRECT (Explicitly map to Command) ===
        ("Kannst du {device} in {area} aufmachen?", "HassTurnOn", {}),
        ("Kannst du {device} in {area} öffnen?", "HassTurnOn", {}),
        ("Kannst du {device} in {area} zumachen?", "HassTurnOff", {}),
        ("Kannst du {device} in {area} schließen?", "HassTurnOff", {}),
        
        # === ABSOLUTE CONTROL (Position) ===
        ("Stelle {device} in {area} auf 50 Prozent", "HassSetPosition", {"position": 50}),
        ("Fahre {device} in {area} auf 50 Prozent", "HassSetPosition", {"position": 50}),
        
        # === RELATIVE CONTROL (Position) ===
        ("Fahre {device} in {area} weiter hoch", "HassSetPosition", {"command": "step_up"}),
        ("Fahre {device} in {area} weiter runter", "HassSetPosition", {"command": "step_down"}),
        ("Mach {device} in {area} weiter auf", "HassSetPosition", {"command": "step_up"}),
        ("Mach {device} in {area} weiter zu", "HassSetPosition", {"command": "step_down"}),
        
        # === TIME-BASED ===
        ("Öffne {device} in {area} in 10 Minuten", "DelayedControl", {"command": "on"}),
        ("Schließe {device} in {area} in 10 Minuten", "DelayedControl", {"command": "off"}),
        ("Öffne {device} in {area} für 10 Minuten", "TemporaryControl", {"command": "on"}),
        # "Close for X minutes" is less common but valid
        ("Schließe {device} in {area} für 10 Minuten", "TemporaryControl", {"command": "off"}),
        
        # === QUERY ===
        ("Ist {device_nom} in {area} offen?", "HassGetState", {"state": "open"}),
        ("Ist {device_nom} in {area} geschlossen?", "HassGetState", {"state": "closed"}),
        ("Sind {device} in {area} offen?", "HassGetState", {"state": "open"}),
    ],
    
    "climate": [
        # === ON/OFF ===
        ("Schalte {device} in {area} an", "HassTurnOn", {}),
        ("Schalte {device} in {area} aus", "HassTurnOff", {}),
        
        # === ABSOLUTE CONTROL (Temperature) ===
        ("Stelle {device} in {area} auf 21 Grad", "HassClimateSetTemperature", {}),
        ("Heize {device} in {area} auf 22 Grad", "HassClimateSetTemperature", {}),
        
        # === RELATIVE CONTROL (Temperature) ===
        ("Mach {device} in {area} wärmer", "HassClimateSetTemperature", {"command": "step_up"}),
        ("Mach {device} in {area} kälter", "HassClimateSetTemperature", {"command": "step_down"}),
        ("Erhöhe die Temperatur in {area}", "HassClimateSetTemperature", {"command": "step_up"}),
        ("Verringere die Temperatur in {area}", "HassClimateSetTemperature", {"command": "step_down"}),
        
        # === TIME-BASED ===
        ("Schalte {device} in {area} in 10 Minuten an", "DelayedControl", {"command": "on"}),
        ("Schalte {device} in {area} in 10 Minuten aus", "DelayedControl", {"command": "off"}),
        ("Heize {area} für 30 Minuten", "TemporaryControl", {"command": "on"}),
        
        # === QUERY ===
        ("Wie warm ist es in {area}?", "HassGetState", {}),
        ("Ist {device_nom} in {area} an?", "HassGetState", {"state": "on"}),
    ],
    
    "fan": [
        # === ON/OFF ===
        ("Schalte {device} in {area} an", "HassTurnOn", {}),
        ("Schalte {device} in {area} aus", "HassTurnOff", {}),
        
        # === ABSOLUTE CONTROL (Percentage) ===
        ("Stelle {device} in {area} auf 50 Prozent", "HassFanSetPercentage", {"percentage": 50}),
        
        # === RELATIVE CONTROL (Speed) ===
        ("Mach {device} in {area} schneller", "HassFanSetPercentage", {"command": "step_up"}),
        ("Mach {device} in {area} langsamer", "HassFanSetPercentage", {"command": "step_down"}),
        
        # === TIME-BASED ===
        ("Schalte {device} in {area} in 10 Minuten an", "DelayedControl", {"command": "on"}),
        ("Schalte {device} in {area} in 10 Minuten aus", "DelayedControl", {"command": "off"}),
        ("Schalte {device} in {area} für 10 Minuten an", "TemporaryControl", {"command": "on"}),
        
        # === QUERY ===
        ("Ist {device_nom} in {area} an?", "HassGetState", {"state": "on"}),
    ],
    
    "media_player": [
        # === ON/OFF ===
        ("Schalte {device} in {area} an", "HassTurnOn", {}),
        ("Schalte {device} in {area} aus", "HassTurnOff", {}),
        
        # === PLAY/PAUSE ===
        ("Pausiere {device} in {area}", "HassMediaPause", {}),
        ("Setze {device} in {area} fort", "HassMediaResume", {}),
        
        # === VOLUME CONTROL (Abs/Rel) ===
        ("Stelle die Lautstärke von {device} in {area} auf 50 Prozent", "HassMediaSetVolume", {"volume_level": 50}),
        ("Mach {device} in {area} lauter", "HassMediaSetVolume", {"command": "volume_up"}),
        ("Mach {device} in {area} leiser", "HassMediaSetVolume", {"command": "volume_down"}),
        
        # === TIME-BASED ===
        ("Schalte {device} in {area} in 10 Minuten aus", "DelayedControl", {"command": "off"}),
        
        # === QUERY ===
        ("Ist {device_nom} in {area} an?", "HassGetState", {"state": "on"}),
    ],
    
    "switch": [
        # === ON/OFF ===
        ("Schalte {device} in {area} an", "HassTurnOn", {}),
        ("Schalte {device} in {area} aus", "HassTurnOff", {}),
        
        # === TIME-BASED ===
        ("Schalte {device} in {area} in 10 Minuten an", "DelayedControl", {"command": "on"}),
        ("Schalte {device} in {area} in 10 Minuten aus", "DelayedControl", {"command": "off"}),
        ("Schalte {device} in {area} für 10 Minuten an", "TemporaryControl", {"command": "on"}),
        
        # === QUERY ===
        ("Ist {device_nom} in {area} an?", "HassGetState", {"state": "on"}),
    ],
    
    "automation": [
        # === ON/OFF ===
        ("Aktiviere {device} in {area}", "HassTurnOn", {}),
        ("Deaktiviere {device} in {area}", "HassTurnOff", {}),
        
        # === TIME-BASED ===
        ("Aktiviere {device} in {area} in 10 Minuten", "DelayedControl", {"command": "on"}),
        ("Deaktiviere {device} in {area} in 10 Minuten", "DelayedControl", {"command": "off"}),
        ("Aktiviere {device} in {area} für 10 Minuten", "TemporaryControl", {"command": "on"}),
        
        # === QUERY ===
        ("Ist {device_nom} in {area} aktiv?", "HassGetState", {"state": "on"}),
    ],
}
