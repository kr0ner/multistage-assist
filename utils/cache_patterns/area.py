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
        ("Schalte {device} {area_prep} {area} an", "HassTurnOn", {}),
        ("Schalte {device} {area_prep} {area} aus", "HassTurnOff", {}),
        ("Mach {device} {area_prep} {area} an", "HassTurnOn", {}),
        ("Mach {device} {area_prep} {area} aus", "HassTurnOff", {}),
        
        # === DIRECT PHRASING (No preposition) ===
        ("{area} {device} an", "HassTurnOn", {}),
        ("{area} {device} aus", "HassTurnOff", {}),
        ("{device} {area} an", "HassTurnOn", {}),
        ("{device} {area} aus", "HassTurnOff", {}),
        ("Licht {area} an", "HassTurnOn", {}), # Manual common case
        ("Licht {area} aus", "HassTurnOff", {}), # Manual common case
        
        # === POLITE / INDIRECT (Explicitly map to Command) ===
        ("Kannst du {device} {area_prep} {area} anschalten?", "HassTurnOn", {}),
        ("Kannst du {device} {area_prep} {area} anmachen?", "HassTurnOn", {}),
        ("Würdest du bitte {device} {area_prep} {area} anmachen?", "HassTurnOn", {}),
        ("Kannst du {device} {area_prep} {area} ausschalten?", "HassTurnOff", {}),
        ("Kannst du {device} {area_prep} {area} ausmachen?", "HassTurnOff", {}),
        
        # === ABSOLUTE CONTROL (Brightness) ===
        ("Stelle {device} {area_prep} {area} auf 50 Prozent", "HassLightSet", {"brightness": 50}),
        ("Setze {device} {area_prep} {area} auf 100 Prozent", "HassLightSet", {"brightness": 100}),
        
        # === RELATIVE CONTROL (Brightness) ===
        ("Mach {device} {area_prep} {area} heller", "HassLightSet:step_up", {"command": "step_up"}),
        ("Mach {device} {area_prep} {area} dunkler", "HassLightSet:step_down", {"command": "step_down"}),
        ("Dimme {device} {area_prep} {area} hoch", "HassLightSet:step_up", {"command": "step_up"}),
        ("Dimme {device} {area_prep} {area} runter", "HassLightSet:step_down", {"command": "step_down"}),
        ("Es ist zu hell {area_prep} {area}", "HassLightSet:step_down", {"command": "step_down"}),
        ("Es ist zu dunkel {area_prep} {area}", "HassLightSet:step_up", {"command": "step_up"}),
        
        # === TIME-BASED ===
        ("Schalte {device} {area_prep} {area} in 10 Minuten an", "DelayedControl", {"command": "on"}),
        ("Schalte {device} {area_prep} {area} in 10 Minuten aus", "DelayedControl", {"command": "off"}),
        ("Schalte {device} {area_prep} {area} für 10 Minuten an", "TemporaryControl", {"command": "on"}),
        ("Schalte {device} {area_prep} {area} für 10 Minuten aus", "TemporaryControl", {"command": "off"}),
        
        # === QUERY ===
        ("Ist {device_nom} {area_prep} {area} an?", "HassGetState", {"state": "on"}),
        ("Sind {device} {area_prep} {area} an?", "HassGetState", {"state": "on"}),
    ],
    
    "cover": [
        # === OPEN/CLOSE (Using HassTurnOn/Off as requested) ===
        # TurnOn = Open, TurnOff = Close
        ("Öffne {device} {area_prep} {area}", "HassTurnOn", {}),
        ("Mach {device} {area_prep} {area} auf", "HassTurnOn", {}),
        ("Fahre {device} {area_prep} {area} hoch", "HassTurnOn", {}),
        
        ("Schließe {device} {area_prep} {area}", "HassTurnOff", {}),
        ("Mach {device} {area_prep} {area} zu", "HassTurnOff", {}),
        ("Fahre {device} {area_prep} {area} runter", "HassTurnOff", {}),
        
        # === POLITE / INDIRECT (Explicitly map to Command) ===
        ("Kannst du {device} {area_prep} {area} aufmachen?", "HassTurnOn", {}),
        ("Kannst du {device} {area_prep} {area} öffnen?", "HassTurnOn", {}),
        ("Kannst du {device} {area_prep} {area} zumachen?", "HassTurnOff", {}),
        ("Kannst du {device} {area_prep} {area} schließen?", "HassTurnOff", {}),
        
        # === ABSOLUTE CONTROL (Position) ===
        ("Stelle {device} {area_prep} {area} auf 50 Prozent", "HassSetPosition", {"position": 50}),
        ("Fahre {device} {area_prep} {area} auf 50 Prozent", "HassSetPosition", {"position": 50}),
        ("Fahre {device} {area_prep} {area} zur Hälfte", "HassSetPosition", {"position": 50}),
        ("Fahre {device} {area_prep} {area} zur Hälfte runter", "HassSetPosition", {"position": 50}),
        ("Fahre {device} {area_prep} {area} zur Hälfte hoch", "HassSetPosition", {"position": 50}),
        ("Mach {device} {area_prep} {area} halb zu", "HassSetPosition", {"position": 50}),
        ("Mach {device} {area_prep} {area} halb auf", "HassSetPosition", {"position": 50}),
        
        ("Fahre {device} {area_prep} {area} ein Viertel", "HassSetPosition", {"position": 25}),
        ("Fahre {device} {area_prep} {area} dreiviertel", "HassSetPosition", {"position": 75}),
        
        # Variants with "Schließe" (common for covers)
        ("Schließe {device} {area_prep} {area} auf 50 Prozent", "HassSetPosition", {"position": 50}),
        ("Schließe {device} {area_prep} {area} zu 50 Prozent", "HassSetPosition", {"position": 50}),
        ("Schließe {device} {area_prep} {area} zur Hälfte", "HassSetPosition", {"position": 50}),
        
        # Variants with "Öffne"
        ("Öffne {device} {area_prep} {area} auf 50 Prozent", "HassSetPosition", {"position": 50}),
        ("Öffne {device} {area_prep} {area} zu 50 Prozent", "HassSetPosition", {"position": 50}),
        ("Öffne {device} {area_prep} {area} zur Hälfte", "HassSetPosition", {"position": 50}),
        
        # "Komplett" = Open/Close (100% / 0%)
        ("Mach {device} {area_prep} {area} ganz auf", "HassTurnOn", {}),
        ("Mach {device} {area_prep} {area} komplett auf", "HassTurnOn", {}),
        ("Mach {device} {area_prep} {area} ganz zu", "HassTurnOff", {}),
        ("Mach {device} {area_prep} {area} komplett zu", "HassTurnOff", {}),
        
        # === RELATIVE CONTROL (Position) ===
        ("Fahre {device} {area_prep} {area} weiter hoch", "HassSetPosition:step_up", {"command": "step_up"}),
        ("Fahre {device} {area_prep} {area} weiter runter", "HassSetPosition:step_down", {"command": "step_down"}),
        ("Mach {device} {area_prep} {area} weiter auf", "HassSetPosition:step_up", {"command": "step_up"}),
        ("Mach {device} {area_prep} {area} weiter zu", "HassSetPosition:step_down", {"command": "step_down"}),
        ("Die Rollläden sind zu weit oben {area_prep} {area}", "HassSetPosition:step_down", {"command": "step_down"}),
        ("Die Rollläden sind zu weit unten {area_prep} {area}", "HassSetPosition:step_up", {"command": "step_up"}),
        
        # === TIME-BASED ===
        ("Öffne {device} {area_prep} {area} in 10 Minuten", "DelayedControl", {"command": "on"}),
        ("Schließe {device} {area_prep} {area} in 10 Minuten", "DelayedControl", {"command": "off"}),
        ("Öffne {device} {area_prep} {area} für 10 Minuten", "TemporaryControl", {"command": "on"}),
        # "Close for X minutes" is less common but valid
        ("Schließe {device} {area_prep} {area} für 10 Minuten", "TemporaryControl", {"command": "off"}),
        
        # === QUERY ===
        ("Ist {device_nom} {area_prep} {area} offen?", "HassGetState", {"state": "open"}),
        ("Ist {device_nom} {area_prep} {area} geschlossen?", "HassGetState", {"state": "closed"}),
        ("Sind {device} {area_prep} {area} offen?", "HassGetState", {"state": "open"}),
    ],
    
    "climate": [
        # === ON/OFF ===
        ("Schalte {device} {area_prep} {area} an", "HassTurnOn", {}),
        ("Schalte {device} {area_prep} {area} aus", "HassTurnOff", {}),
        
        # === ABSOLUTE CONTROL (Temperature) ===
        ("Stelle {device} {area_prep} {area} auf 21 Grad", "HassClimateSetTemperature", {}),
        ("Heize {device} {area_prep} {area} auf 22 Grad", "HassClimateSetTemperature", {}),
        
        # === RELATIVE CONTROL (Temperature) ===
        ("Mach {device} {area_prep} {area} wärmer", "HassClimateSetTemperature:step_up", {"command": "step_up"}),
        ("Mach {device} {area_prep} {area} kälter", "HassClimateSetTemperature:step_down", {"command": "step_down"}),
        ("Erhöhe die Temperatur {area_prep} {area}", "HassClimateSetTemperature:step_up", {"command": "step_up"}),
        ("Verringere die Temperatur {area_prep} {area}", "HassClimateSetTemperature:step_down", {"command": "step_down"}),
        
        # === TIME-BASED ===
        ("Schalte {device} {area_prep} {area} in 10 Minuten an", "DelayedControl", {"command": "on"}),
        ("Schalte {device} {area_prep} {area} in 10 Minuten aus", "DelayedControl", {"command": "off"}),
        ("Heize {area} für 30 Minuten", "TemporaryControl", {"command": "on"}),
        
        # === QUERY ===
        ("Wie warm ist es {area_prep} {area}?", "HassGetState", {}),
        ("Ist {device_nom} {area_prep} {area} an?", "HassGetState", {"state": "on"}),
    ],
    
    "fan": [
        # === ON/OFF ===
        ("Schalte {device} {area_prep} {area} an", "HassTurnOn", {}),
        ("Schalte {device} {area_prep} {area} aus", "HassTurnOff", {}),
        
        # === ABSOLUTE CONTROL (Percentage) ===
        ("Stelle {device} {area_prep} {area} auf 50 Prozent", "HassFanSetPercentage", {"percentage": 50}),
        
        # === RELATIVE CONTROL (Speed) ===
        ("Mach {device} {area_prep} {area} schneller", "HassFanSetPercentage:step_up", {"command": "step_up"}),
        ("Mach {device} {area_prep} {area} langsamer", "HassFanSetPercentage:step_down", {"command": "step_down"}),
        
        # === TIME-BASED ===
        ("Schalte {device} {area_prep} {area} in 10 Minuten an", "DelayedControl", {"command": "on"}),
        ("Schalte {device} {area_prep} {area} in 10 Minuten aus", "DelayedControl", {"command": "off"}),
        ("Schalte {device} {area_prep} {area} für 10 Minuten an", "TemporaryControl", {"command": "on"}),
        
        # === QUERY ===
        ("Ist {device_nom} {area_prep} {area} an?", "HassGetState", {"state": "on"}),
    ],
    
    "media_player": [
        # === ON/OFF ===
        ("Schalte {device} {area_prep} {area} an", "HassTurnOn", {}),
        ("Schalte {device} {area_prep} {area} aus", "HassTurnOff", {}),
        
        # === PLAY/PAUSE ===
        ("Pausiere {device} {area_prep} {area}", "HassMediaPause", {}),
        ("Setze {device} {area_prep} {area} fort", "HassMediaResume", {}),
        
        # === VOLUME CONTROL (Abs/Rel) ===
        ("Stelle die Lautstärke von {device} {area_prep} {area} auf 50 Prozent", "HassMediaSetVolume", {"volume_level": 50}),
        ("Mach {device} {area_prep} {area} lauter", "HassMediaSetVolume:volume_up", {"command": "volume_up"}),
        ("Mach {device} {area_prep} {area} leiser", "HassMediaSetVolume:volume_down", {"command": "volume_down"}),
        ("Musik {area_prep} {area} ist zu laut", "HassMediaSetVolume:volume_down", {"command": "volume_down"}),
        ("Musik {area_prep} {area} ist zu leise", "HassMediaSetVolume:volume_up", {"command": "volume_up"}),
        
        # === TIME-BASED ===
        ("Schalte {device} {area_prep} {area} in 10 Minuten aus", "DelayedControl", {"command": "off"}),
        
        # === QUERY ===
        ("Ist {device_nom} {area_prep} {area} an?", "HassGetState", {"state": "on"}),
    ],
    
    "switch": [
        # === ON/OFF ===
        ("Schalte {device} {area_prep} {area} an", "HassTurnOn", {}),
        ("Schalte {device} {area_prep} {area} aus", "HassTurnOff", {}),
        
        # === TIME-BASED ===
        ("Schalte {device} {area_prep} {area} in 10 Minuten an", "DelayedControl", {"command": "on"}),
        ("Schalte {device} {area_prep} {area} in 10 Minuten aus", "DelayedControl", {"command": "off"}),
        ("Schalte {device} {area_prep} {area} für 10 Minuten an", "TemporaryControl", {"command": "on"}),
        
        # === QUERY ===
        ("Ist {device_nom} {area_prep} {area} an?", "HassGetState", {"state": "on"}),
    ],
    
    "automation": [
        # === ON/OFF ===
        ("Aktiviere {device} {area_prep} {area}", "HassTurnOn", {}),
        ("Deaktiviere {device} {area_prep} {area}", "HassTurnOff", {}),
        
        # === TIME-BASED ===
        ("Aktiviere {device} {area_prep} {area} in 10 Minuten", "DelayedControl", {"command": "on"}),
        ("Deaktiviere {device} {area_prep} {area} in 10 Minuten", "DelayedControl", {"command": "off"}),
        ("Aktiviere {device} {area_prep} {area} für 10 Minuten", "TemporaryControl", {"command": "on"}),
        
        # === QUERY ===
        ("Ist {device_nom} {area_prep} {area} aktiv?", "HassGetState", {"state": "on"}),
    ],
}
