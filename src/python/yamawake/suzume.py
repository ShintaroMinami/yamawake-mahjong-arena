"""suzume: the bot decided inside Rust (`src/rust/suzume/`) by per-decision MLPs over
universal features, fitted to a reinforcement-learned teacher.

    python -m yamawake.suzume list        # every rule, in evaluation order

Seat it with `agents.SuzumeAgent()` or the player spec `suzume` (also
`suzume-rla225`; `agents.SUZUME_SETS`): these sets take no options. The legacy
sets (`legacy-suzume-mlp3[:KEY=VALUE...]` and the others of `agents.LEGACY_SETS`)
take these options (`KNOBS`; everything else is fixed with the models):

    kan            0/1 (default 1): make kans when the kan MLP prefers them
    call           0/1 (default 1): chi and pon when the call MLP prefers them
    riichi         0/1 (default 1): declare riichi when the riichi MLP prefers it
    fold           0/1 (default 1): betaori against a riichi (only with pf=0)
    fold_shanten   0..8 (default 1): the best after-discard shanten from which
                   `fold` folds
    pf             0/1 (default 1): push / fold by the predicted push / mawashi /
                   fold state (replaces `fold`)

e.g. `--players legacy-suzume-mlp3:call=0 rulebot`.
"""

from __future__ import annotations

import argparse

from yamawake._core import suzume_config, suzume_rules

#: The options a suzume player may set.
KNOBS = tuple(suzume_config({}))


def rules() -> list[dict]:
    """Every rule in registry (evaluation) order.

    Returns:
        Dicts with `name`, `mode`, `decision` and `description`.
    """
    return suzume_rules()


def effective_config(config: dict | None = None) -> dict:
    """The options the bot runs with (raises ValueError on an unknown option or a bad value)."""
    return suzume_config(dict(config or {}))


def _table() -> str:
    lines = [f"{'#':>2}  {'name':<16} {'mode':<5} {'decision':<8} description"]
    for i, r in enumerate(rules()):
        lines.append(f"{i:>2}  {r['name']:<16} {r['mode']:<5} {r['decision']:<8} {r['description']}")
    lines.append("")
    lines.append("options (default): " + ", ".join(f"{k}={v}" for k, v in effective_config().items()))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """`python -m yamawake.suzume list`."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("list",))
    ap.parse_args(argv)
    print(_table())


if __name__ == "__main__":
    main()
