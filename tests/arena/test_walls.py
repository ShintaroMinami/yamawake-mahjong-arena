"""The standard walls regenerate exactly (their fingerprints: walls.check_fingerprints)."""
from __future__ import annotations

import pytest

from yamawake.walls import (HONBA, KYOKU, TILES, blob_for_wall, check_fingerprints,
                             generate_matrix, standard_walls)


def test_all_5000_standard_walls_match_their_fingerprints():
    std = standard_walls()
    assert std["walls"] == 5000 and std["default_walls"] == 1000
    assert check_fingerprints() == {
        "fingerprint_first": True, "fingerprint_last": True,
        "fingerprint_first500_sha256": True, "fingerprint_all_sha256": True}


def test_a_matrix_is_permutations_and_packs_to_138_byte_records():
    mat = generate_matrix(123)
    assert mat.shape == (KYOKU, HONBA, TILES)
    assert all(sorted(mat[k, h]) == list(range(TILES))
               for k in range(KYOKU) for h in range(HONBA))
    blob = blob_for_wall(123)
    assert len(blob) == KYOKU * HONBA * 138 and blob[:2] == bytes((0, 0))


def test_the_loop_rejects_a_malformed_wall_table():
    from yamawake import SelfPlayLoop
    with pytest.raises(ValueError, match="multiple of 138"):
        SelfPlayLoop([1], 2, wall_tables=[b"\x00" * 137])
    bad = bytearray(blob_for_wall(5))
    bad[3] = bad[2]                                  # a tile twice
    with pytest.raises(ValueError, match="permutation"):
        SelfPlayLoop([1], 2, wall_tables=[bytes(bad)])
