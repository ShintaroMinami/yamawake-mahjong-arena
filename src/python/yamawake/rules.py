"""The settlement rule: uma, oka and ranks.

    pt_i = ((final_score_i - return_points) / 1000 + uma_eff[rank_i]) / scale

`uma_eff` is the rank bonus with the oka (the start/return gap x4) pooled to
first place, so the game is zero-sum. The default is Tenhou's: uma
+90/+45/0/-135 (in 1000s, with the oka 20 already on top: +110 for first),
25,000 start, 30,000 return, `scale` 100.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TENHOU_UMA = (90.0, 45.0, 0.0, -135.0)


def rank_of(scores: list[int]) -> np.ndarray:
    """Final ranks (0 = first) with the seat-order tiebreak (earlier seat wins)."""
    order = sorted(range(4), key=lambda s: (-scores[s], s))
    ranks = np.empty(4, dtype=np.int64)
    for r, s in enumerate(order):
        ranks[s] = r
    return ranks


@dataclass(frozen=True)
class UmaRule:
    """One table rule: uma, start/return points and the pt scale.

    `name` is the label a match records as its `protocol["rule"]`. It stays
    "RewardRule" -- the name results were recorded under before yamawake
    split out -- so old and new results compare field for field.
    """

    uma: tuple[float, float, float, float] = TENHOU_UMA
    start_points: int = 25000
    return_points: int = 30000
    scale: float = 100.0
    name: str = "RewardRule"

    @property
    def uma_eff(self) -> np.ndarray:
        """Rank bonus with the oka pooled to first place."""
        u = np.asarray(self.uma, dtype=np.float32).copy()
        u[0] += 4 * (self.return_points - self.start_points) / 1000.0
        return u

    def rank_of(self, scores: list[int]) -> np.ndarray:
        """Final ranks (0 = first); see the module-level `rank_of`."""
        return rank_of(scores)

    def final_pt(self, scores: list[int] | np.ndarray) -> np.ndarray:
        """Normalized final match points per seat."""
        scores = np.asarray(scores, dtype=np.float32)
        ranks = rank_of([int(s) for s in scores])
        pt = (scores - self.return_points) / 1000.0 + self.uma_eff[ranks]
        return (pt / self.scale).astype(np.float32)


DEFAULT_RULE = UmaRule()


def rule_label(rule) -> str:
    """The label a match records for `rule`: its `name`, else its class name."""
    return getattr(rule, "name", None) or type(rule).__name__
