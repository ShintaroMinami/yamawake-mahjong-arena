"""The rulebot's frozen configs, as knob dicts for the in-loop Rust bot.

`STANDARD` is the rulebot's standard config (the player `rulebot`): `LATEST`
with the call assessment switched to the single-player DP pricing
(statistically equal strength) and ankan / kakan by kobalab's
`select_gang` (`kan`, never daiminkan).
`LATEST` (the player `rulebot:latest`) is the last config of the config sweeps
(kobalab_ev call pricing). Seat one with `agents.RulebotAgent(STANDARD)`; the bot is
decided inside the Rust loop (`src/rust/rulebot/`). The knobs are only what
tells the two apart (`RulebotConfig` in `src/rust/rulebot/config.rs`); any knob
not given keeps the Rust default, and an unknown knob is an error.
"""

from __future__ import annotations

LATEST: dict = dict(call_ev_sp=False, call_sp_flat_pass=False, kan=False)

STANDARD: dict = dict(LATEST, call_ev_sp=True, call_sp_flat_pass=True, kan=True)
