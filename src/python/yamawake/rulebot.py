"""The rulebot's frozen configs, as knob dicts for the in-loop Rust bot.

`STANDARD` is the rulebot's standard config (the player `rulebot`): `KOBALAB`
with the call assessment switched to the single-player DP pricing
(statistically equal strength) and ankan / kakan by kobalab's
`select_gang` (`kan`, never daiminkan).
`KOBALAB` (the player `rulebot:kobalab`) prices calls with kobalab_ev (the
port of majiang-ai's hand evaluation) and declares no kan; `STANDARD` was built
on it. Seat one with `agents.RulebotAgent(STANDARD)`; the bot is decided inside
the Rust loop (`src/rust/rulebot/`). The knobs are only what
tells the two apart (`RulebotConfig` in `src/rust/rulebot/config.rs`); any knob
not given keeps the Rust default (on, as in `STANDARD`), and an unknown knob is
an error.
"""

from __future__ import annotations

KOBALAB: dict = dict(call_ev_sp=False, call_sp_flat_pass=False, kan=False)

STANDARD: dict = dict(KOBALAB, call_ev_sp=True, call_sp_flat_pass=True, kan=True)
