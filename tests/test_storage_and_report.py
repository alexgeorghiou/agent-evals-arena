from evals_arena.arena import compare_runs, decide
from evals_arena.elo import Leaderboard
from evals_arena.judge import HeuristicPairwiseJudge, PairwiseJudge
from evals_arena.loader import load_suite
from evals_arena.providers import create_provider
from evals_arena.report import arena_markdown, leaderboard_markdown, run_markdown
from evals_arena.results import CaseResult
from evals_arena.runner import Runner
from evals_arena.storage import ResultStore


def run_weather(suites_dir, model):
    suite = load_suite(suites_dir / "weather-agent.yaml")
    provider = create_provider(f"fake:{model}", fake_script_paths=[suites_dir])
    return suite, Runner(provider).run(suite)


def test_store_round_trip_and_listing(tmp_path, suites_dir):
    store = ResultStore(tmp_path)
    _, first = run_weather(suites_dir, "fake-strong")
    _, second = run_weather(suites_dir, "fake-sloppy")
    store.save(first)
    store.save(second)
    loaded = store.load(first.run_id)
    assert loaded.summary == first.summary
    assert loaded.cases[0].trajectory == first.cases[0].trajectory
    assert {r["model"] for r in store.list()} == {"fake-strong", "fake-sloppy"}
    assert store.list(model="fake-sloppy")[0]["run_id"] == second.run_id
    assert store.latest("weather-agent", "fake-strong").run_id == first.run_id
    assert store.load(first.run_id[:-3]).run_id == first.run_id  # unique prefix


def test_run_markdown_lists_failures(suites_dir):
    _, run = run_weather(suites_dir, "fake-sloppy")
    md = run_markdown(run)
    assert "`weather-agent` on `fake-sloppy`" in md
    assert "`paris-now`" in md
    assert "Failing cases" in md


class AlwaysTie(PairwiseJudge):
    def prefer(self, **kwargs):
        return "tie"


def case(score, error=None, output="x"):
    return CaseResult(case_id="c", passed=score > 0.5, score=score, output=output, error=error)


def test_decide_rules():
    judge = AlwaysTie()
    assert decide(judge, "t", case(1.0, error="boom"), case(0.1), None).winner == "B"
    assert decide(judge, "t", case(0.9), case(0.2), None).decided_by == "assertions"
    assert decide(judge, "t", case(0.9), case(0.85), None).winner == "tie"


def test_compare_runs_updates_leaderboard(tmp_path, suites_dir):
    suite, strong = run_weather(suites_dir, "fake-strong")
    _, sloppy = run_weather(suites_dir, "fake-sloppy")
    board = Leaderboard(tmp_path / "lb.sqlite")
    report = compare_runs(suite, strong, sloppy, HeuristicPairwiseJudge(), board)
    assert report.tally()["A"] >= 4
    assert report.rating_a > report.rating_b
    assert "fake-strong" in arena_markdown(report)
    assert "| 1 | `fake-strong` |" in leaderboard_markdown(board.standings())
