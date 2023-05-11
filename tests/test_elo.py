import pytest

from evals_arena.elo import DEFAULT_RATING, Leaderboard, expected_score, update


def test_expected_score_properties():
    assert expected_score(1000, 1000) == pytest.approx(0.5)
    assert expected_score(1400, 1000) == pytest.approx(10 / 11)
    assert expected_score(1200, 1000) + expected_score(1000, 1200) == pytest.approx(1)


def test_update_is_zero_sum_and_directional():
    a, b = update(1000, 1000, "A", k=32)
    assert (a, b) == (pytest.approx(1016), pytest.approx(984))
    a, b = update(1200, 1000, "tie", k=32)
    assert a < 1200 < a + 32
    assert a + b == pytest.approx(2200)


def test_upset_moves_more_than_expected_win():
    gain_expected = update(1400, 1000, "A")[0] - 1400
    gain_upset = update(1000, 1400, "A")[0] - 1000
    assert gain_upset > gain_expected > 0


def test_leaderboard_records_matches(tmp_path):
    board = Leaderboard(tmp_path / "lb.sqlite", k=16)
    board.record("s", "c1", "m1", "m2", "A")
    board.record("s", "c2", "m1", "m2", "tie")
    board.record("s", "c1", "m3", "m1", "B")
    standings = {s.model: s for s in board.standings()}
    assert standings["m1"].games == 3
    assert (standings["m1"].wins, standings["m1"].ties) == (2, 1)
    assert standings["m2"].losses == 1
    assert standings["m3"].rating < DEFAULT_RATING
    assert board.standings()[0].model == "m1"
    assert sum(s.rating for s in standings.values()) == pytest.approx(3 * DEFAULT_RATING)
    assert len(board.matches()) == 3


def test_leaderboard_rejects_self_play(tmp_path):
    with pytest.raises(ValueError, match="itself"):
        Leaderboard(tmp_path / "lb.sqlite").record("s", "c", "m", "m", "A")
