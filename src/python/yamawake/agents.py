"""What can sit at an arena seat, and how a player spec names it.

An agent answers decisions; how it turns a decision into features is its own
business. Two lanes exist, and an agent declares which one it wants:

    LANE_OBSERVATION   `act_batch(decisions) -> actions`, the general lane:
                       the agent gets the situation and returns a move
    LANE_RULEBOT       decided inside Rust (`config` is the knob dict);
                       nothing reaches Python
    LANE_SUZUME       the same for suzume (`config` is its option dict)

One call per tick, never one per decision: every game in flight waiting on this
agent arrives in one list, which is what lets a model batch its forward.

An observation agent may also carry the two context hooks of `context.py`:
`reducer_factory` (a per-event reducer each of its trackers keeps) and `annotate`
(attached to each of its decisions under `Decision.extra`).

Player specs (`build_agent`): `tsumogiri`, `rulebot`, `rulebot:latest`,
`suzume[:KEY=VALUE[:KEY=VALUE...]]` (e.g. `suzume:call=0`; `suzume` is
`suzume-mlp3`, and `suzume-mlp3-flat`, `suzume-mlp3-avoid4`, `suzume-mlp1`
and `suzume-mlp1-flat` are the other model sets, `MODEL_SETS`), plus
whatever other packages register -- by name (`register_agent`), by a resolver
for open-ended specs such as checkpoint paths (`register_resolver`), or through
the `yamawake.agents` entry-point group, whose callables are run once (with no
arguments) to register.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional, Protocol, runtime_checkable

import riichienv as rv

from yamawake._core import suzume_model_sets
from yamawake.context import Decision

LANE_OBSERVATION = "observation"
LANE_RULEBOT = "rulebot"
LANE_SUZUME = "suzume"


@runtime_checkable
class ArenaAgent(Protocol):
    """The player protocol. `lane` picks how the arena feeds it."""

    lane: str

    def act_batch(self, decisions: list[Decision]) -> list[rv.Action]:
        """One action per decision, in order."""

    def describe(self) -> dict:
        """Who this is, for the result and the recorded log's names."""


class TsumogiriAgent:
    """Discards the drawn tile, passes on everything else."""

    lane = LANE_OBSERVATION

    def act_batch(self, decisions: list[Decision]) -> list[rv.Action]:
        """Tsumogiri where possible, else pass, else the first legal action."""
        return [_tsumogiri(d.obs) for d in decisions]

    def describe(self) -> dict:
        """`{"kind": "tsumogiri", "name": "tsumogiri"}`."""
        return {"kind": "tsumogiri", "name": "tsumogiri"}


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


#: The suzume model sets, each its own player name (`src/rust/suzume/params.rs`,
#: `ModelSet`: a hand model set from `params/hand/` and a state model from
#: `params/state/`); `suzume` is the default set, `suzume-mlp3`.
MODEL_SETS = ("suzume", *suzume_model_sets())


class SuzumeAgent:
    """The suzume bot (`src/rust/suzume/`): decided inside the Rust loop."""

    lane = LANE_SUZUME

    def __init__(self, config: dict | None = None, name: str = "suzume", models: str = "suzume") -> None:
        """Args: config: option dict (`yamawake.suzume` lists them; an unknown
        knob raises ValueError here); name: its label; models: the model set
        (`MODEL_SETS`)."""
        from yamawake._core import suzume_config
        if models not in MODEL_SETS:
            raise ValueError(f"unknown suzume model set {models!r} (sets: {', '.join(MODEL_SETS)})")
        if "models" in (config or {}):
            raise ValueError("the model set is picked by the player name, not an option")
        self.config = dict(config or {})
        if models != "suzume":
            self.config["models"] = models
        suzume_config(self.config)
        self.name = name

    def describe(self) -> dict:
        """`{"kind": "suzume", "name": <name>, "config": <options>}`."""
        return {"kind": "suzume", "name": self.name, "config": dict(self.config)}


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
        spec: The player spec (`tsumogiri`, `rulebot`, `rulebot:latest`,
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
            or `suzume-mlp1:call=0`.
        obs_b64: The seat's riichienv `Observation`, `serialize_to_base64()`.
        kyoku_events: The kyoku's mjai events so far, from its `start_kyoku`
            (JSON lines or dicts; `Decision.kyoku_events` is one).
        debug: Also return the rulebot's branch inputs.

    Returns:
        `{"action": (action_type, tile, consume_tiles)}`, plus `rule` (the
        deciding suzume rule) or, with `debug`, `debug` (rulebot). A suzume
        bot starts from an empty kyoku memo (no push / fold state carried
        from its earlier decisions).

    Raises:
        ValueError: The spec is not a rulebot or suzume bot.
    """
    from yamawake._core import decide_one
    agent = build_agent(spec)
    kind = {LANE_RULEBOT: "rulebot", LANE_SUZUME: "suzume"}.get(agent.lane)
    if kind is None:
        raise ValueError(f"{spec!r} is not a rulebot or suzume bot")
    events = [e if isinstance(e, str) else json.dumps(e) for e in kyoku_events]
    return decide_one(kind, agent.config, obs_b64, events, debug)


def _rulebot(spec: str, **_) -> RulebotAgent:
    from yamawake.rulebot import STANDARD, LATEST
    configs = {"rulebot": STANDARD, "rulebot:latest": LATEST}
    if spec not in configs:
        raise ValueError(f"unknown rulebot config: {spec}")
    return RulebotAgent(configs[spec], name=spec)


def _suzume(spec: str, **options) -> SuzumeAgent:
    """`<model set>[:KEY=VALUE...]` (`MODEL_SETS`); `--opt` options that are
    suzume options apply too (the spec's own pairs win)."""
    from yamawake.suzume import KNOBS
    config = {k: v for k, v in options.items() if k in KNOBS}
    models, *pairs = spec.split(":")
    for pair in pairs:
        k, sep, v = pair.partition("=")
        if not sep:
            raise ValueError(f"suzume spec takes KEY=VALUE pairs, got {pair!r} in {spec!r}")
        config[k] = v
    return SuzumeAgent(config, name=spec, models=models)


register_agent("tsumogiri", lambda spec, **_: TsumogiriAgent())
register_agent("rulebot", _rulebot)
for _set in MODEL_SETS:
    register_agent(_set, _suzume)
