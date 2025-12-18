# Vacuum Control

Control robot vacuums with room-specific and mode-specific commands.

## Commands

| Command | Mode | Area |
|---------|------|------|
| "Sauge die Küche" | vacuum | Küche |
| "Wische den Keller" | mop | Keller |
| "Staubsauge das Bad" | vacuum | Bad |
| "Feuchtwischen im Flur" | mop | Flur |

## Mode Detection

**Vacuum mode** (dry cleaning):
- saugen, sauge
- staubsaugen, staubsauge
- absaugen

**Mop mode** (wet cleaning):
- wischen, wische
- nass, feucht
- moppen, feuchtwischen

## Floor-Level Commands

| Command | What Happens |
|---------|--------------|
| "Sauge das Erdgeschoss" | Cleans ground floor |
| "Wische das Obergeschoss" | Mops upper floor |

## Global Commands

| Command | Scope |
|---------|-------|
| "Sauge das ganze Haus" | GLOBAL |
| "Staubsauge überall" | GLOBAL |

## Article Removal

German articles are automatically stripped:
- "den Keller" → "Keller"
- "die Küche" → "Küche"
- "das Bad" → "Bad"
