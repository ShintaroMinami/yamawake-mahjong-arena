"""What can sit at an arena seat, and how a player spec names it.

An agent answers decisions; how it turns a decision into features is its own
business. Two lanes exist, and an agent declares which one it wants:

    LANE_OBSERVATION   `act_batch(decisions) -> actions`, the general lane:
                       the agent gets the situation and returns a move
    LANE_RULEBOT       decided inside Rust (`config` is the knob dict);
                       nothing reaches Python
    LANE_SUZUME       the same for the legacy suzume sets (`config` is its
                       option dict)
    LANE_TSUMOGIRI     the same for the tsumogiri bot (no options)
    LANE_A1            the same for suzume's A1 sets (`config`: {"a1": a
                       built-in set's name}, or {"a1": a model directory, the
                       rest the suzume config of its features set} as
                       suzume-train plays its fits)

One call per tick, never one per decision: every game in flight waiting on this
agent arrives in one list, which is what lets a model batch its forward.

An observation agent may also carry the two context hooks of `context.py`:
`reducer_factory` (a per-event reducer each of its trackers keeps) and `annotate`
(attached to each of its decisions under `Decision.extra`).

Player specs (`build_agent`): `tsumogiri` (in Rust), `tsumogiri-python` (the
same moves through the observation lane), `rulebot`, `rulebot:kobalab`,
`suzume[:KEY=VALUE...]` (the default set; `STYLE_OPTIONS` or `STYLE_TARGETS`) and
`suzume-rla225[:KEY=VALUE...]` (`SUZUME_SETS`; `STYLE_OPTIONS`),
the legacy sets `legacy-suzume-mlp3[:KEY=VALUE[:KEY=VALUE...]]` (e.g.
`legacy-suzume-mlp3:call=0`; also `legacy-suzume-mlp3-flat`, `legacy-suzume-mlp3-avoid4`,
`legacy-suzume-mlp1`, `legacy-suzume-mlp1-flat`, `LEGACY_SETS`), plus
whatever other packages register -- by name (`register_agent`), by a resolver
for open-ended specs such as checkpoint paths (`register_resolver`), or through
the `yamawake.agents` entry-point group, whose callables are run once (with no
arguments) to register.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional, Protocol, runtime_checkable

import riichienv as rv

from yamawake._core import a1_model_sets, suzume_model_sets
from yamawake.context import Decision
from yamawake.style import STYLE_TARGETS

LANE_OBSERVATION = "observation"
LANE_RULEBOT = "rulebot"
LANE_SUZUME = "suzume"
LANE_TSUMOGIRI = "tsumogiri"
LANE_A1 = "a1"
#: The lanes decided inside the Rust loop.
IN_LOOP_LANES = (LANE_RULEBOT, LANE_SUZUME, LANE_TSUMOGIRI, LANE_A1)


@runtime_checkable
class ArenaAgent(Protocol):
    """The player protocol. `lane` picks how the arena feeds it."""

    lane: str

    def act_batch(self, decisions: list[Decision]) -> list[rv.Action]:
        """One action per decision, in order."""

    def describe(self) -> dict:
        """Who this is, for the result and the recorded log's names."""


class TsumogiriAgent:
    """Discards the drawn tile, passes on everything else: decided inside
    the Rust loop (`arena::tsumogiri`), never in Python."""

    lane = LANE_TSUMOGIRI
    config: dict = {}

    def describe(self) -> dict:
        """`{"kind": "tsumogiri", "name": "tsumogiri"}`."""
        return {"kind": "tsumogiri", "name": "tsumogiri"}


class PythonTsumogiriAgent:
    """The tsumogiri bot's moves made in Python, on the observation lane: a
    minimal observation-lane agent, and a gauge of the Python lane's cost
    against `TsumogiriAgent`."""

    lane = LANE_OBSERVATION

    def act_batch(self, decisions: list[Decision]) -> list[rv.Action]:
        """Tsumogiri where possible, else pass, else the first legal action."""
        return [_tsumogiri(d.obs) for d in decisions]

    def describe(self) -> dict:
        """`{"kind": "tsumogiri-python", "name": "tsumogiri-python"}`."""
        return {"kind": "tsumogiri-python", "name": "tsumogiri-python"}


def _tsumogiri(obs: rv.Observation) -> rv.Action:
    legal = obs.legal_actions()
    drawn_tile = obs.drawn_tile
    for a in legal:
        if a.action_type == rv.ActionType.DISCARD and a.tile == drawn_tile:
            return a
    for a in legal:
        if a.action_type == rv.ActionType.PASS:
            return a
    return legal[0]


class RulebotAgent:
    """The ported rulebot: decided inside the Rust loop, never in Python."""

    lane = LANE_RULEBOT

    def __init__(self, config: dict | None = None, name: str = "rulebot") -> None:
        """Args: config: knob dict (default `rulebot.STANDARD`); name: its label."""
        from yamawake.rulebot import STANDARD
        self.config = dict(config or STANDARD)
        self.name = name

    def describe(self) -> dict:
        """`{"kind": "rulebot", "name": <name>}`."""
        return {"kind": "rulebot", "name": self.name}


#: The suzume sets, each its own player name: A1 sets (`src/rust/suzume/a1.rs`,
#: `BUILTIN`, `params/suzume/`), the default `suzume` first; they take no options.
SUZUME_SETS = tuple(a1_model_sets())
#: The legacy suzume sets until 2026-10-21 (`src/rust/suzume/params.rs`,
#: `ModelSet`: a hand model set from `params/legacy/hand/` and a state model
#: from `params/legacy/state/`); they take the suzume options.
LEGACY_SETS = tuple(suzume_model_sets())
MODEL_SETS = (*SUZUME_SETS, *LEGACY_SETS)
#: The play-style options of an A1 set (2026-10-24): `aggression` (push / fold, -1 .. 1; needs the set's endpoint
#: models, which only `suzume` has) and `call_shift` / `riichi_shift` (-1 .. 1; scaled score offsets,
#: `src/rust/suzume/a1.rs`, `A1Config`).
STYLE_OPTIONS = ("aggression", "call_shift", "riichi_shift")
#: The style coordinates `suzume` also takes in their place (2026-10-31, `yamawake.style`): call_rate, riichi_rate,
#: push_z (`STYLE_TARGETS`); calibrated into `STYLE_OPTIONS` when the agent is built, a coordinate left out at
#: default suzume's value.
#: The set the style coordinates are calibrated for.
STYLE_TARGET_SET = "suzume"


class SuzumeAgent:
    """The suzume bot (`src/rust/suzume/`): decided inside the Rust loop.

    A set of `SUZUME_SETS` plays on the A1 lane (`config` `{"a1": <set>}`), a
    legacy set (`LEGACY_SETS`, or `dir:<path>`) on the suzume lane.
    """

    def __init__(self, config: dict | None = None, name: str = "suzume", models: str = "suzume") -> None:
        """Args: config: option dict: for a legacy set the suzume options
        (`yamawake.suzume` lists them; an unknown knob raises ValueError here),
        for an A1 set `STYLE_OPTIONS` (checked when the bot is built); name: its label;
        models: the model set (`MODEL_SETS`), or `dir:<path>` for a legacy-layout
        set read from a directory (for fitting new models;
        `src/rust/suzume/params.rs`, `load_dir`)."""
        from yamawake._core import suzume_config
        if models not in MODEL_SETS and not models.startswith("dir:"):
            raise ValueError(f"unknown suzume model set {models!r} (sets: {', '.join(MODEL_SETS)})")
        if "models" in (config or {}):
            raise ValueError("the model set is picked by the player name, not an option")
        self.name = name
        self.style = None
        if models in SUZUME_SETS:
            config = dict(config or {})
            allowed = STYLE_OPTIONS + (STYLE_TARGETS if models == STYLE_TARGET_SET else ())
            bad = [k for k in config if k not in allowed]
            if bad:
                raise ValueError(f"{models} takes only {', '.join(allowed)} (got {', '.join(bad)}); the legacy sets "
                                 "take the suzume options")
            if any(k in config for k in STYLE_TARGETS):
                config = self._calibrate(config)
            self.lane, self.config = LANE_A1, {"a1": models, **config}
            return
        self.lane = LANE_SUZUME
        self.config = {**(config or {}), "models": models}
        suzume_config(self.config)

    def _calibrate(self, config: dict) -> dict:
        """The style coordinates in `config` -> `STYLE_OPTIONS` (`yamawake.style.calibrate`); warns when the
        target is out of reach or needs options past the fitted grid."""
        import warnings
        from yamawake.style import calibrate, default_style
        if any(k in config for k in STYLE_OPTIONS):
            raise ValueError(f"give either the style coordinates ({', '.join(STYLE_TARGETS)}) or the options "
                             f"({', '.join(STYLE_OPTIONS)}), not both")
        try:
            target = {**default_style(), **{k: float(v) for k, v in config.items()}}
        except ValueError:
            raise ValueError(f"style coordinates must be numbers, got {config}") from None
        res = calibrate(*(target[k] for k in STYLE_TARGETS))
        self.style = {"target": target, **res}
        if not res["reached"]:
            got = ", ".join(f"{k} {res['predicted'][k]:.3f}" for k in STYLE_TARGETS)
            warnings.warn(f"style target {', '.join(f'{k} {target[k]:.3f}' for k in STYLE_TARGETS)} is out of reach; "
                          f"playing the nearest ({got})", stacklevel=3)
        elif res["extrapolated"]:
            warnings.warn(f"style target needs options past the calibrated -1 .. 1 ({res['options']}); "
                          "the style it plays is an extrapolation", stacklevel=3)
        return res["options"]

    def describe(self) -> dict:
        """`{"kind": "suzume", "name": <name>, "config": <options>}`, plus `style` (target, options,
        predicted style, reached) when built from style coordinates."""
        out = {"kind": "suzume", "name": self.name, "config": dict(self.config)}
        if self.style:
            out["style"] = self.style
        return out


# --- the player-spec registry -----------------------------------------------

#: A factory builds an agent from the full spec and the caller's options.
AgentFactory = Callable[..., Any]
#: A resolver returns an agent for a spec it recognizes, else None.
AgentResolver = Callable[..., Optional[Any]]

_FACTORIES: dict[str, AgentFactory] = {}
_RESOLVERS: list[AgentResolver] = []
_PLUGINS_LOADED = False
ENTRY_POINT_GROUP = "yamawake.agents"


def register_agent(name: str, factory: AgentFactory) -> None:
    """Make `name` (and `name:<arg>`) a player spec.

    Args:
        name: The spec, or its part before the first ":".
        factory: `factory(spec, **options) -> agent`; it gets the whole spec,
            so `name:<arg>` can carry an argument.
    """
    _FACTORIES[name] = factory


def register_resolver(resolver: AgentResolver) -> None:
    """Add a fallback for specs no name matches (e.g. checkpoint paths).

    Args:
        resolver: `resolver(spec, **options) -> agent | None`; tried in
            registration order after the names.
    """
    _RESOLVERS.append(resolver)


def _load_plugins() -> None:
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    _PLUGINS_LOADED = True
    from importlib.metadata import entry_points
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        ep.load()()


def build_agent(spec: str, **options):
    """Resolve one player spec into an agent.

    Args:
        spec: The player spec (`tsumogiri`, `rulebot`, `rulebot:kobalab`,
            `suzume[:KEY=VALUE...]`, or one a plugin registered).
        **options: Passed to the factory or resolver (e.g. `device=` for a
            model player); the built-in agents take none.

    Returns:
        An `ArenaAgent`.

    Raises:
        ValueError: The spec names nothing.
    """
    _load_plugins()
    head = spec.split(":", 1)[0]
    for key in (spec, head):
        if key in _FACTORIES:
            return _FACTORIES[key](spec, **options)
    for resolver in _RESOLVERS:
        agent = resolver(spec, **options)
        if agent is not None:
            return agent
    raise ValueError(f"unknown player: {spec}")


def decide(spec: str, obs_b64: str, kyoku_events: list, debug: bool = False) -> dict:
    """One decision of an in-loop bot on a position, as the arena's loop makes it.

    Args:
        spec: A rulebot or suzume player spec (`build_agent`), e.g. `rulebot`
            or `legacy-suzume-mlp1:call=0`.
        obs_b64: The seat's riichienv `Observation`, `serialize_to_base64()`.
        kyoku_events: The kyoku's mjai events so far, from its `start_kyoku`
            (JSON lines or dicts; `Decision.kyoku_events` is one).
        debug: Also return the rulebot's branch inputs.

    Returns:
        `{"action": (action_type, tile, consume_tiles)}`, plus `rule` (the
        deciding suzume rule; `a1` for a set of `SUZUME_SETS`) or, with `debug`, `debug` (rulebot). A suzume
        bot starts from an empty kyoku memo (no push / fold state carried
        from its earlier decisions).

    Raises:
        ValueError: The spec is not a rulebot or suzume bot.
    """
    from yamawake._core import decide_one
    agent = build_agent(spec)
    if agent.lane not in IN_LOOP_LANES:
        raise ValueError(f"{spec!r} is not an in-loop bot (rulebot, suzume, tsumogiri)")
    events = [e if isinstance(e, str) else json.dumps(e) for e in kyoku_events]
    return decide_one(agent.lane, agent.config, obs_b64, events, debug)


def _rulebot(spec: str, **_) -> RulebotAgent:
    from yamawake.rulebot import STANDARD, KOBALAB
    configs = {"rulebot": STANDARD, "rulebot:kobalab": KOBALAB}
    if spec not in configs:
        raise ValueError(f"unknown rulebot config: {spec}")
    return RulebotAgent(configs[spec], name=spec)


def _suzume(spec: str, **options) -> SuzumeAgent:
    """`<model set>[:KEY=VALUE...]` (`MODEL_SETS`); `--opt` options that are
    the set's options (suzume options, or `STYLE_OPTIONS` for an A1 set) apply too
    (the spec's own pairs win)."""
    from yamawake.suzume import KNOBS
    models, *pairs = spec.split(":")
    keys = STYLE_OPTIONS + (STYLE_TARGETS if models == STYLE_TARGET_SET else ()) if models in SUZUME_SETS else KNOBS
    config = {k: v for k, v in options.items() if k in keys}
    for pair in pairs:
        k, sep, v = pair.partition("=")
        if not sep:
            raise ValueError(f"suzume spec takes KEY=VALUE pairs, got {pair!r} in {spec!r}")
        config[k] = v
    return SuzumeAgent(config, name=spec, models=models)


register_agent("tsumogiri", lambda spec, **_: TsumogiriAgent())
register_agent("tsumogiri-python", lambda spec, **_: PythonTsumogiriAgent())
register_agent("rulebot", _rulebot)
for _set in MODEL_SETS:
    register_agent(_set, _suzume)
