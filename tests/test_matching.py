import pytest

from evals_arena.matching import explain_mismatch, matches


@pytest.mark.parametrize(
    ("expected", "actual"),
    [
        ("paris", "Paris"),
        ("par*", "Paris"),
        ("re:^o(slo)?$", "Oslo"),
        ("<any>", None),
        ({"city": "paris"}, {"city": "Paris", "unit": "celsius"}),
        ({"opts": {"days": 3}}, {"opts": {"days": 3.0, "x": 1}}),
        ([1, "a*"], [1, "abc"]),
        ("42", 42),
        (True, True),
    ],
)
def test_matches(expected, actual):
    assert matches(expected, actual)


@pytest.mark.parametrize(
    ("expected", "actual"),
    [
        ("paris", "Paris, TX"),
        ({"city": "paris"}, {"town": "paris"}),
        ([1, 2], [1, 2, 3]),
        (True, 1),
        (1, True),
        ({"a": 1}, "not a dict"),
        ("re:^x$", "xy"),
    ],
)
def test_does_not_match(expected, actual):
    assert not matches(expected, actual)


def test_explain_mismatch_points_at_path():
    assert explain_mismatch({"a": {"b": 1}}, {"a": {"b": 2}}) == "$.a.b: expected 1, got 2"
    assert explain_mismatch({"a": 1}, {}) == "$.a missing"
    assert explain_mismatch({"a": 1}, {"a": 1}) is None
