# Upcoming event data

The active source is `upcoming_events.json`. It is intentionally empty until
real, verified fight-card data is entered. The application reloads this file
on every events request, so normal card updates require no code change or
server restart.

Use this shape:

```json
{
  "events": [
    {
      "event_id": "stable-event-id",
      "event_name": "Published event name",
      "event_date": "2026-12-31",
      "location": "Published location or null",
      "fights": [
        {
          "fight_id": "stable-fight-id",
          "fighter_a": "Exact published fighter name",
          "fighter_b": "Exact published fighter name",
          "weight_class": "Lightweight",
          "number_of_rounds": 5,
          "bout_order": 1,
          "card_section": "Main Event"
        }
      ]
    }
  ]
}
```

Only real, verified cards should be entered. Unknown or insufficient-history
fighters receive an unavailable result rather than substituted statistics.
