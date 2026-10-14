"""Explicit (kyoku, honba) wall matrices for duplicate evaluation.

A wall matrix pins the wall DATA each game-progress coordinate plays on,
where a generator seed's assignment drifts with play (renchan shifts the
kyoku ordinal a seeded engine indexes by). One matrix covers every
reachable coordinate of a hanchan -- KYOKU=16 rounds (E1..N4) x
HONBA=12 repeats -- and is shared by every
rotation of a duplicate pair, so combined with the engine's split-wall
mode (per-seat draw queues) all players experience the same wall
opportunities regardless of calls and renchan.

Routine runs share the standard set (`standard_walls()`, 5,000 matrices
from one recorded base id), so two results differ by their players only;
`fresh_wall_ids` draws a set never used before. Ids are deterministic
handles (record them, e.g. a run's base id, to regenerate a run's
walls); adjacent ids share nothing (SeedSequence-hashed streams).

Blob format consumed by ``yamawake._core.SelfPlayLoop(wall_tables=...)``:
concatenated 138-byte records ``(kyoku_ordinal, honba, 136 tiles)``,
each tile list a permutation of 0..135. Coordinates beyond the matrix
(honba >= 12) fall back to the engine's seeded shuffle, which stays
deterministic per game seed.
"""

from __future__ import annotations

import json
from importlib import resources

import numpy as np

KYOKU = 16   # E1..E4, S1..S4, W1..W4, N1..N4
HONBA = 12
TILES = 136
_SEED_NS = 0x52454C4A  # "RELJ": keeps matrix streams away from other rngs
#: Ask `run_match` for a never-used wall set instead of the standard one.
FRESH = "fresh"


def generate_matrix(wall_id: int, kyoku: int = KYOKU,
                    honba: int = HONBA) -> np.ndarray:
    """Generate one frozen wall matrix, shape (kyoku, honba, 136), uint8.

    Deterministic in `wall_id` alone (PCG64 stream seeded by the id; the
    cells are drawn sequentially from that one stream). `wall_id` is an
    arbitrary identifier -- adjacent ids share nothing.

    Args:
        wall_id: Identifier of the matrix (any int; record it to be able
            to regenerate the same walls).
        kyoku: Number of kyoku rows (default E1..N4).
        honba: Honba depth per kyoku (default 12).

    Returns:
        Array of independent 136-tile permutations.
    """
    rng = np.random.Generator(np.random.PCG64([_SEED_NS, wall_id]))
    mat = np.empty((kyoku, honba, TILES), dtype=np.uint8)
    for m in range(kyoku):
        for n in range(honba):
            mat[m, n] = rng.permutation(TILES).astype(np.uint8)
    return mat


def matrix_blob(mat: np.ndarray) -> bytes:
    """Pack a wall matrix into the SelfPlayLoop `wall_tables` blob."""
    kyoku, honba, tiles = mat.shape
    if tiles != TILES:
        raise ValueError(f"wall matrix last dim must be {TILES}, got {tiles}")
    out = bytearray()
    for m in range(kyoku):
        for n in range(honba):
            out += bytes((m, n)) + mat[m, n].tobytes()
    return bytes(out)


def blob_for_wall(wall_id: int) -> bytes:
    """The packed blob of one wall id (generate + pack)."""
    return matrix_blob(generate_matrix(wall_id))


def fresh_wall_ids(walls: int,
                   base_id: int | None = None) -> tuple[int, list[int]]:
    """Fresh wall ids for one evaluation run.

    Draws a random 48-bit base id (or reuses a recorded one) and returns
    ``(base_id, [base_id + w, ...])``: fresh walls every run, yet any
    run's walls regenerate exactly from its logged base_id.
    """
    import secrets

    if base_id is None:
        base_id = secrets.randbits(48)
    return base_id, [base_id + w for w in range(walls)]


def standard_walls() -> dict:
    """The recorded standard wall set (package data `eval_walls.json`).

    Keys: `wall_base`, `walls` (5,000), `default_walls` (1,000) and the four
    fingerprints `check_fingerprints` verifies. A run takes the FIRST `walls`
    of the set, so a 1,000-wall run is a prefix of a 5,000-wall one.
    """
    text = resources.files("yamawake").joinpath("data/eval_walls.json").read_text()
    return json.loads(text)


def fingerprints(wall_base: int, walls: int) -> dict[str, str]:
    """The four fingerprints of `walls` walls from `wall_base`.

    first / last: sha256 of one wall's blob; first500 / all: sha256 of the
    concatenated per-wall hex digests, in wall order.
    """
    import hashlib

    hexs = [hashlib.sha256(blob_for_wall(wall_base + w)).hexdigest() for w in range(walls)]

    def agg(hs: list[str]) -> str:
        return hashlib.sha256("".join(hs).encode()).hexdigest()

    return {"fingerprint_first": hexs[0], "fingerprint_last": hexs[-1],
            "fingerprint_first500_sha256": agg(hexs[:500]),
            "fingerprint_all_sha256": agg(hexs)}


def check_fingerprints() -> dict[str, bool]:
    """Regenerate every standard wall and compare its four fingerprints."""
    std = standard_walls()
    got = fingerprints(std["wall_base"], std["walls"])
    return {k: v == std[k] for k, v in got.items()}
