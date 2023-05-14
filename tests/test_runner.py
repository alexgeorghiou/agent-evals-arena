import pytest

from evals_arena.judge import HeuristicJudge
from evals_arena.loader import load_suites
from evals_arena.pricing import Price, PriceTable
from evals_arena.providers import Usage, create_provider
from evals_arena.runner import Runner
from evals_arena.stats import bootstrap_ci, percentile

from .conftest import SUITES


@pytest.fixture(scope="module")
def suites():
    return [s for _, s in load_suites([SUITES])]


@pytest.mark.parametrize("model", ["fake-strong", "fake-sloppy"])
def test_example_suites_separate_good_and_bad_agents(suites, model):
    provider = create_provider(f"fake:{model}", fake_script_paths=[SUITES])
    runner = Runner(provider, judge=HeuristicJudge())
    runs = [runner.run(s) for s in suites]
    pass_rates = [r.summary.pass_rate for r in runs]
    if model == "fake-strong":
        assert pass_rates == [1.0] * len(runs)
    else:
        assert max(pass_rates) == 0.0


def test_concurrency_preserves_order_and_results(suites):
    provider = create_provider("fake:fake-strong", fake_script_paths=[SUITES])
    serial = Runner(provider, judge=HeuristicJudge()).run(suites[0])
    parallel = Runner(provider, judge=HeuristicJudge(), concurrency=4).run(suites[0])
    assert [c.case_id for c in serial.cases] == [c.case_id for c in parallel.cases]
    assert [c.score for c in serial.cases] == [c.score for c in parallel.cases]


def test_tag_filter(suites):
    weather = next(s for s in suites if s.name == "weather-agent")
    run = Runner(create_provider("fake:fake-strong"), tags={"smoke"}).run(weather)
    assert [c.case_id for c in run.cases] == ["paris-now"]


def test_cost_tracking_uses_price_table(suites):
    weather = next(s for s in suites if s.name == "weather-agent")
    provider = create_provider("fake:fake-strong", fake_script_paths=[SUITES])
    prices = PriceTable({"fake-strong": Price(1_000_000, 0)})  # $1 per input token
    run = Runner(provider, prices=prices).run(weather)
    for case in run.cases:
        assert case.cost_usd == pytest.approx(case.input_tokens)
    assert run.summary.total_cost_usd == pytest.approx(run.summary.input_tokens)
    assert run.summary.latency_p95_ms >= run.summary.latency_p50_ms


def test_price_lookup_prefix_and_overrides(tmp_path):
    table = PriceTable()
    assert table.lookup("claude-opus-5-5") == Price(4.0, 20.0)
    assert table.lookup("gpt-4.1-mini-2026") == Price(0.4, 1.6)  # longest prefix wins
    assert table.cost("unknown-model", Usage(10, 10)) == 0.0
    overrides = tmp_path / "prices.yaml"
    overrides.write_text("my-model: {input: 1, output: 2}\n")
    loaded = PriceTable.load(overrides)
    assert loaded.cost("my-model", Usage(1_000_000, 500_000)) == pytest.approx(2.0)


def test_percentiles_and_ci():
    values = [100, 200, 300, 400, 1000]
    assert percentile(values, 50) == 300
    assert percentile(values, 95) == pytest.approx(880)
    assert percentile([], 50) == 0
    with pytest.raises(ValueError, match="pct"):
        percentile(values, 101)
    low, high = bootstrap_ci([0.0, 1.0] * 10)
    assert 0 <= low < 0.5 < high <= 1
    assert bootstrap_ci([0.4]) == (0.4, 0.4)
