# Calendar Events

Create calendar events using natural language with multi-turn conversations.

## Basic Usage

```
User: "Erstelle einen Termin morgen um 15 Uhr"
Assistant: "Wie soll der Termin heißen?"
User: "Zahnarzt"
Assistant: [preview] "Sag 'Ja' zum Bestätigen."
User: "Ja"
Assistant: "Termin wurde erstellt."
```

## Information Gathered

1. **Summary** (required) - Event title
2. **Date/Time** (required) - When
3. **Calendar** (required if multiple exposed) - Which calendar
4. **Duration** (optional) - Event length

## Relative Date Patterns

| Pattern | Example |
|---------|---------|
| heute | Today |
| morgen | Tomorrow |
| übermorgen | Day after tomorrow |
| in X Tagen | "in 37 Tagen" → 37 days from now |
| X Tage | "37 Tage" → 37 days from now |
| nächsten [Wochentag] | Next Monday, etc. |
| am [Wochentag] | Next occurrence |

## Time Ranges

```
"Termin morgen von 15 Uhr bis 18 Uhr"
→ Start: 15:00, End: 18:00
```

## Calendar Selection

- **One calendar exposed**: Auto-selected
- **Multiple calendars**: System asks "In welchen Kalender?"
- Only calendars exposed to conversation assistant are shown

## Confirmation Flow

Before creating, shows preview:
```
Termin erstellen?
📅 **Event Name**
🕐 DD.MM.YYYY um HH:MM Uhr
📁 Kalender: Calendar Name
```

Confirm with: ja, ok, genau, richtig, passt
Cancel with: nein, abbrechen, stop
