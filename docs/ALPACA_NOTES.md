# Alpaca (Paper) Mode — Notes and Replit Tips

This project focuses on a general 15m strategy with MT5 live execution, but you may also run a sibling bot with Alpaca (paper) — e.g., on Replit.

## Setup (Paper)

- Create an Alpaca paper account
- Generate API keys (paper)
- Environment variables required:
  - `ALPACA_API_KEY`, `ALPACA_API_SECRET`
  - Optional: `ACCOUNT_START_EQUITY` (for FTMO-style guardrails)
- Paper endpoint in code:
  - `BASE_URL = 'https://paper-api.alpaca.markets'`

## Common startup issue: get_position 404

Alpaca returns a 404 APIError when no position exists for a symbol. Treat this as “no position” rather than a fatal error.

Recommended wrapper:

```python
from alpaca_trade_api.rest import APIError

def safe_get_position(api, symbol: str):
    try:
        return api.get_position(symbol)
    except APIError as e:
        msg = str(e).lower()
        if getattr(e, 'status_code', None) == 404 or 'position does not exist' in msg:
            return None
        raise
```

Then in reconciliation:

```python
pos = safe_get_position(self.api, STOCK_TO_TRADE)
if pos is None:
    # no live position; ensure state reflects flat
    self.state.update({
        'in_position': False,
        'position_qty': 0,
        'stop_order_id': None,
    })
    save_state(self.state)
else:
    # map live position to state as needed
    ...
```

## State helpers

If running on Replit or ephemeral environments, keep JSON state simple and robust:

```python
import json, os

def load_state(path='bot_state.json'):
    if os.path.exists(path):
        with open(path, 'r') as f:
            try:
                return json.load(f)
            except Exception:
                pass
    return {
        'in_position': False,
        'position_qty': 0,
        'trade_high_price': 0.0,
        'entry_atr': 0.0,
        'entry_price': 0.0,
        'entry_time': None,
        'partial_profit_taken': False,
        'edge_buy_streak': 0,
        'edge_exit_streak': 0,
        'last_trade_ts': None,
        'trades_today': 0,
        'trades_day_str': None,
        'stop_order_id': None,
        'loss_streak': 0,
        'circuit_breaker_until': None,
        'account_start_equity': None,
    }

def save_state(state, path='bot_state.json'):
    with open(path, 'w') as f:
        json.dump(state, f)
```

## Replit tips

- Use a 15-minute loop; sleep until the next bar boundary to avoid double-processing
- Market-closed handling: poll more slowly (e.g., 30 minutes) outside RTH
- Log to both console and rotating file handler
- Keep risk small (e.g., 0.25%–0.50% per trade) and respect daily stop logic on paper accounts

## Results & portfolio screenshot

Place a PNG under `docs/media/alpaca_portfolio.png` and embed it in README:

```markdown
![Alpaca Portfolio](docs/media/alpaca_portfolio.png)
```

Use the repo README to summarize policies (risk caps, sessions, ML gate) and include a small, labeled chart. Recruiters like a clean screenshot paired with 2–3 bullets that explain the controls.

***

If you want, we can add a minimal `live/trading_bot_alpaca.py` adapter that mirrors the MT5 logic with Alpaca’s REST API, using the same sessions, risk caps, and trailing/partials.
