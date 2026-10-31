"""Target play style -> suzume's style options (2026-10-31).

The style coordinates are those of the paper (`paper/style_calib/calibration.md`):

    call_rate    kyoku with a call made / kyoku where a call (chi, pon, open kan) was legal
    riichi_rate  kyoku with riichi declared / kyoku where riichi was legal
    push_z       pushing under another seat's riichi: z = (ln D - ln D0) / SIGMA, where D is the mean
                 number of deal-in combinations of the tiles cut (own riichi discards included);
                 0 = default suzume, one unit = the spread of Tenhou Houou players

Each rate is a few-parameter structural model of (aggression, call_shift, riichi_shift), fitted on a
729-point grid of 1v3 games against three default suzume (package data `style_calibration.json`; forms
as `paper/code_cog2027/analysis/style_calib/calibrate.py`, `StructModel`). `calibrate` solves the
three equations by least squares on the fitting scales (logit / log), in units of each rate's SE over
4,000 hanchan, with a weak pull of the options toward 0, as the paper's style reproduction did.

    >>> calibrate(0.364, 0.670, -0.78)["options"]   # the Houou median
    {'aggression': -0.379, 'call_shift': 0.253, 'riichi_shift': -0.011}
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources

import numpy as np

#: The style coordinates, in the order `calibrate` takes them.
STYLE_TARGETS = ("call_rate", "riichi_rate", "push_z")
#: push_z = (ln D - ln D0) / SIGMA: D0 is default suzume's D, SIGMA the spread of Houou players.
D0, SIGMA = 3.2529, 0.158
#: Each rate's SE over 4,000 hanchan on its fitting scale: the unit of the residuals.
SE4000 = np.array([0.0112, 0.025, 0.0119])
#: Prior SD of the options: a weak pull toward 0 where a rate hardly moves with an option.
PRIOR_SD = 1.0
#: The search box of (aggression, call_shift, riichi_shift).
LOWER, UPPER = np.array([-1.0, -4.0, -4.0]), np.array([1.0, 4.0, 4.0])
_STARTS = [(0, 0, 0), (0.5, 0.5, 0), (-0.5, 0.5, 0.3), (0.3, -0.5, 0.5)]
_MAIN = ("call_take_kyoku", "riichi_take_kyoku", "danger_comb_mean_all")


@lru_cache(maxsize=1)
def _models() -> dict:
    text = resources.files("yamawake").joinpath("data/style_calibration.json").read_text()
    spec = json.loads(text)["models"]
    return {k: spec[k] for k in _MAIN}


def _sig(v: np.ndarray) -> np.ndarray:
    return 0.5 * (1 + np.tanh(v / 2))


def _logit(v: float) -> float:
    return float(np.log(v / (1 - v)))


def predict(aggression: float, call_shift: float, riichi_shift: float) -> dict[str, float]:
    """The style the structural models predict for one setting (against three default suzume).

    Args:
        aggression: Push / fold option.
        call_shift: Call option.
        riichi_shift: Riichi option.

    Returns:
        `{"call_rate", "riichi_rate", "push_z"}`.
    """
    call, riichi, d = _rates(np.array([aggression, call_shift, riichi_shift], dtype=float))
    return {"call_rate": float(call), "riichi_rate": float(riichi), "push_z": float((np.log(d) - np.log(D0)) / SIGMA)}


def _rates(p: np.ndarray) -> tuple[float, float, float]:
    """(call rate, riichi rate, D) on the natural scale."""
    a, c, r = p
    ms = _models()
    ap, am = max(a, 0.0), max(-a, 0.0)
    uc = 3.4 * c if c >= 0 else 2.5 * c                 # the score offsets (`STYLE_SCALES` in a1.rs)
    ur = 3.5 * r if r >= 0 else 4.5 * r
    # call: a distribution function in u_c (slope, main step, small step near u_c = -1)
    q = ms["call_take_kyoku"]["p"]
    poly = lambda c4: c4[0] + c4[1] * am + c4[2] * am ** 2 + c4[3] * ap
    n1 = q[7] + q[8] * am ** 2 + q[9] * ap
    call = (q[0] + n1 * uc + q[2] * ur + q[1] * (1 + q[3] * ur) * _sig((uc - poly(q[10:14])) / np.exp(poly(q[14:18])))
            + q[4] * _sig((uc - q[5]) / np.exp(q[6])))
    # riichi: logit of Legendre terms in x_r plus small x_c and aggression terms
    xc, xr = uc / 3.4, ur / 4.5
    row = np.r_[np.polynomial.legendre.legvander([xr], 5)[0], xc, xc ** 2, xc * xr, xc * xr ** 2, a, a ** 2]
    riichi = 1 / (1 + np.exp(-row @ np.array(ms["riichi_take_kyoku"]["beta"])))
    # push: Legendre terms in t (aggression, the fold side scaled by K) and couplings with x_c, x_r
    spec = ms["danger_comb_mean_all"]
    t = spec["K"] * a if a < 0 else a
    tm = min(t, 0.0)
    row = np.r_[np.polynomial.legendre.legvander([t], 3)[0], xc, xr, t * xc, t * xr, xc ** 2, xr ** 2, xc * xr,
                t * xc ** 2, t * xr ** 2, t ** 2 * xc, t ** 2 * xr, xc ** 3, xr ** 3, t * xc ** 3, t * xr ** 3,
                tm * xc, tm * xr]
    d = row @ np.array(spec["beta"])
    return float(call), float(riichi), float(d)


def _fitting_scale(rates: tuple[float, float, float]) -> np.ndarray:
    call, riichi, d = rates
    return np.array([_logit(np.clip(call, 1e-6, 1 - 1e-6)), _logit(np.clip(riichi, 1e-6, 1 - 1e-6)),
                     np.log(max(d, 1e-6))])


def _solve(fun, start: np.ndarray, iters: int = 200) -> np.ndarray:
    """Levenberg-Marquardt inside [LOWER, UPPER] (steps clipped to the box), numerical Jacobian."""
    p, lam = np.clip(np.asarray(start, dtype=float), LOWER, UPPER), 1e-3
    f = fun(p)
    for _ in range(iters):
        J = np.column_stack([(fun(p + h) - fun(p - h)) / 2e-6 for h in np.eye(3) * 1e-6])
        g, H = J.T @ f, J.T @ J
        while True:
            q = np.clip(p - np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-12), g), LOWER, UPPER)
            fq = fun(q)
            if fq @ fq < f @ f:
                break
            lam *= 4
            if lam > 1e12:
                return p
        step, p, f, lam = np.abs(q - p).max(), q, fq, lam / 3
        if step < 1e-10:
            break
    return p


def calibrate(call_rate: float, riichi_rate: float, push_z: float) -> dict:
    """suzume's style options whose predicted style best matches the target.

    Args:
        call_rate: Target share of the kyoku where a call was legal in which suzume calls (0 .. 1).
        riichi_rate: Target share of the kyoku where riichi was legal in which suzume declares (0 .. 1).
        push_z: Target pushing against riichi (0 = default suzume, +1 = one Houou-player SD bolder).

    Returns:
        `{"options": {"aggression", "call_shift", "riichi_shift"} (3 decimals), "predicted": the style
        those options give, "residual_se": each rate's miss in SEs of 4,000 hanchan, "reached": every
        miss within 1, "extrapolated": an option past the fitted grid's -1 .. 1 (untested there)}`. An
        unreachable target returns the nearest setting with `reached` False.

    Raises:
        ValueError: A rate outside (0, 1) or a non-finite target.
    """
    if not all(np.isfinite(v) for v in (call_rate, riichi_rate, push_z)):
        raise ValueError(f"style target ({call_rate}, {riichi_rate}, {push_z}): not finite")
    if not (0 < call_rate < 1 and 0 < riichi_rate < 1):
        raise ValueError(f"call_rate {call_rate} / riichi_rate {riichi_rate}: rates must be in (0, 1)")
    target = np.array([_logit(call_rate), _logit(riichi_rate), np.log(D0) + SIGMA * push_z])
    miss = lambda p: (_fitting_scale(_rates(p)) - target) / SE4000
    fun = lambda p: np.r_[miss(p), p / PRIOR_SD]
    p = min((_solve(fun, np.array(s, dtype=float)) for s in _STARTS), key=lambda x: fun(x) @ fun(x))
    p = np.round(p, 3) + 0.0                             # 3 decimals, as the paper's player specs; no -0.0
    res = miss(p)
    return {"options": dict(zip(("aggression", "call_shift", "riichi_shift"), p.tolist())),
            "predicted": predict(*p), "residual_se": res.tolist(), "reached": bool(np.abs(res).max() < 1),
            "extrapolated": bool(np.abs(p).max() > 1)}


def default_style() -> dict[str, float]:
    """Default suzume's style as the models give it (all options 0)."""
    return predict(0.0, 0.0, 0.0)
