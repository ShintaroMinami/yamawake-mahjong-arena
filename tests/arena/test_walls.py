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


def test_seeded_walls_are_per_seed_and_index_and_apart_from_ids():
    from yamawake.walls import seeded_matrix
    a = seeded_matrix(7, 0)
    assert a.shape == (KYOKU, HONBA, TILES) and (a == seeded_matrix(7, 0)).all()
    assert all(sorted(a[k, h]) == list(range(TILES)) for k in range(KYOKU) for h in range(HONBA))
    others = [seeded_matrix(7, 1), seeded_matrix(8, 0), seeded_matrix(0, 7), generate_matrix(7),
              generate_matrix((7 << 32) | 0)]
    assert all((a != o).mean() > 0.9 for o in others)


@pytest.mark.parametrize("seed, index", [(-1, 0), (2 ** 32, 0), (0, -1), (0, 2 ** 32)])
def test_seeded_walls_reject_values_outside_32_bits(seed, index):
    from yamawake.walls import seeded_matrix
    with pytest.raises(ValueError, match=r"\[0, 2\*\*32\)"):
        seeded_matrix(seed, index)
