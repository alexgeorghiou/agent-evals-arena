import pytest

from evals_arena.gate import Baseline, GatePolicy, gate, gate_run
from evals_arena.report import gate_markdown
from evals_arena.results import CaseResult, RunResult


def make_run(scores, *, model="m", suite="s", cost=0.0, latency=100.0, threshold=0.7):
    cases = [
        CaseResult(
            case_id=f"c{i}",
            passed=s >= threshold,
            score=s,
            output="",
            cost_usd=cost / len(scores),
            latency_ms=latency,
        )
        for i, s in enumerate(scores)
    ]
    return RunResult(run_id="r", suite=suite, model=model, provider="fake", cases=cases).finalise()


@pytest.fixture
def baseline():
    b = Baseline()
    b.record(make_run([1.0, 1.0, 0.5, 0.9], cost=0.01))
    return b


def test_unchanged_run_passes(baseline):
    result = gate([make_run([1.0, 1.0, 0.5, 0.9], cost=0.01)], baseline, GatePolicy())
    assert result.passed
    assert result.suites[0].score_delta == pytest.approx(0)


def test_score_drop_beyond_threshold_fails(baseline):
    res = gate_run(make_run([1.0, 0.92, 0.5, 0.9]), baseline.get("s", "m"), GatePolicy())
    assert res.status == "pass"  # drop of 0.02 is within the default allowance
    res = gate_run(make_run([0.8, 0.8, 0.5, 0.9]), baseline.get("s", "m"), GatePolicy())
    assert res.status == "fail"
    assert any("score dropped" in r for r in res.reasons)


def test_case_regression_fails_even_when_mean_holds(baseline):
    # c0 drops below threshold, c2 improves by the same amount -> mean unchanged
    run = make_run([0.6, 1.0, 0.9, 0.9])
    res = gate_run(run, baseline.get("s", "m"), GatePolicy(max_pass_rate_drop=1))
    assert res.regressed_cases == ["c0"]
    assert res.improved_cases == ["c2"]
    assert res.status == "fail"
    relaxed = GatePolicy(max_pass_rate_drop=1, fail_on_case_regression=False)
    assert gate_run(run, baseline.get("s", "m"), relaxed).status == "pass"


def test_cost_and_latency_budgets(baseline):
    run = make_run([1.0, 1.0, 0.5, 0.9], cost=0.02, latency=500)
    assert gate_run(run, baseline.get("s", "m"), GatePolicy()).status == "pass"
    strict = GatePolicy(max_cost_increase_pct=50, max_latency_increase_pct=100)
    res = gate_run(run, baseline.get("s", "m"), strict)
    assert res.status == "fail"
    assert len(res.reasons) == 2


def test_new_suite_is_reported_not_failed(baseline):
    res = gate([make_run([0.2], suite="fresh")], baseline, GatePolicy())
    assert res.passed
    assert res.suites[0].status == "new"
    floor = gate([make_run([0.2], suite="fresh")], baseline, GatePolicy(min_score=0.5))
    assert not floor.passed


def test_gate_against_other_models_baseline(baseline):
    res = gate(
        [make_run([0.1, 0.1, 0.1, 0.1], model="candidate")],
        baseline,
        GatePolicy(),
        baseline_model="m",
    )
    assert not res.passed
    assert res.suites[0].model == "candidate"


def test_baseline_round_trip(tmp_path, baseline):
    path = tmp_path / "b" / "baseline.json"
    baseline.save(path)
    loaded = Baseline.load(path)
    assert loaded.get("s", "m") == baseline.get("s", "m")
    assert Baseline.load(tmp_path / "missing.json").suites == {}


def test_gate_markdown_lists_reasons(baseline):
    md = gate_markdown(gate([make_run([0.1, 0.1, 0.1, 0.1])], baseline, GatePolicy()))
    assert "Eval gate FAILED" in md
    assert "| FAIL | `s` | `m` |" in md
    assert "cases now failing" in md
