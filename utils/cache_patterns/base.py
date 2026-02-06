"""Shared constants and helpers for semantic cache patterns."""

# Device words with articles for pattern generation
# Key: domain, Value: device word (singular)
# Used for: "Schalte die [Lampe] an"
DOMAIN_DEVICE_WORDS_SINGULAR = {
    "light": "das Licht",
    "switch": "den Schalter",
    "sensor": "den Sensor",
    "cover": "den Rollladen",
    "climate": "die Heizung",
    "fan": "den Ventilator",
    "media_player": "das Gerät",
    "vacuum": "den Staubsauger",
    "automation": "die Automatisierung",
}

# Plural/Group device words for Area patterns
# Used for: "Schalte die [Lichter] im Büro an"
DOMAIN_DEVICE_WORDS = {
    "light": "die Lichter",
    "switch": "die Schalter",
    "sensor": "die Sensoren",
    "cover": "die Rollläden",
    "climate": "die Heizungen",
    "fan": "die Ventilatoren",
    "media_player": "die Geräte",
    "vacuum": "die Staubsauger",
    "automation": "die Automatisierungen",
}

# Nominative case for questions ("Ist [der Rollladen] offen?")
# Default is same as singular if not specified
# Nominative case for questions ("Ist [der Rollladen] offen?")
DOMAIN_DEVICE_WORDS_NOMINATIVE = {
    "light": "das Licht",
    "switch": "der Schalter",
    "sensor": "der Sensor",
    "cover": "der Rollladen",
    "climate": "die Heizung",
    "fan": "der Ventilator",
    "media_player": "das Gerät",
    "vacuum": "der Staubsauger",
    "automation": "die Automatisierung",
}

# Dative case for "von" phrases ("Helligkeit von [dem Licht]")
DOMAIN_DEVICE_WORDS_DATIVE = {
    "light": "dem Licht",
    "switch": "dem Schalter",
    "sensor": "dem Sensor",
    "cover": "dem Rollladen",
    "climate": "der Heizung",
    "fan": "dem Ventilator",
    "media_player": "dem Gerät",
    "vacuum": "dem Staubsauger",
    "automation": "der Automatisierung",
}
