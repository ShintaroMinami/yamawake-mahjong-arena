"""yamawake: a duplicate riichi mahjong arena.

Every wall is a fixed (kyoku, honba) matrix played through the seat
rotations, with a split draw queue per seat, so the luck of the draw cancels
between the players compared.

    from yamawake import RulebotAgent, TsumogiriAgent, run_match

    print(run_match([TsumogiriAgent(), RulebotAgent(), RulebotAgent(),
                     RulebotAgent()], walls=50).table())

`SuzumeAgent` seats suzume, the bot decided inside Rust by per-decision MLPs
fitted to a reinforcement-learned teacher (`yamawake.suzume` lists its rules
and knobs).

Low level: `SelfPlayLoop` (the Rust lockstep loop), `SeatTracker` /
`decision_of` (decision context), `mjai_action_spec` (the answer format),
`decide` (one rulebot / suzume decision on a position),
`walls.blob_for_wall` (the wall tables).
"""

from yamawake._core import SelfPlayLoop, suzume_rules, run_sp
from yamawake.agents import (LANE_SUZUME, LANE_OBSERVATION, LANE_RULEBOT,
                              LANE_TSUMOGIRI, ArenaAgent, PythonTsumogiriAgent,
                              SuzumeAgent, RulebotAgent,
                              TsumogiriAgent, build_agent, decide,
                              register_agent, register_resolver)
from yamawake.context import Decision, SeatTracker, decision_of
from yamawake.driver import mjai_action_spec, play_games
from yamawake.match import (MatchResult, SlotResult, lineup, run_match)
from yamawake.rules import DEFAULT_RULE, UmaRule, rank_of
from yamawake.walls import FRESH, blob_for_wall, standard_walls

__all__ = ["DEFAULT_RULE", "FRESH", "LANE_SUZUME", "LANE_OBSERVATION",
           "LANE_RULEBOT", "LANE_TSUMOGIRI", "ArenaAgent", "PythonTsumogiriAgent", "Decision", "MatchResult",
           "SuzumeAgent", "RulebotAgent", "SeatTracker", "SelfPlayLoop",
           "SlotResult", "TsumogiriAgent", "UmaRule", "blob_for_wall",
           "build_agent", "decide", "decision_of", "lineup", "mjai_action_spec",
           "suzume_rules", "play_games", "rank_of",
           "register_agent", "register_resolver", "run_match", "run_sp",
           "standard_walls"]
