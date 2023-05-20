import json

from typer.testing import CliRunner

from evals_arena.cli import app

from .conftest import SUITES

runner = CliRunner()


def invoke(*args):
    return runner.invoke(app, [str(a) for a in args])


def test_validate_bundled_suites():
    result = invoke("validate", SUITES)
    assert result.exit_code == 0, result.output
    assert "weather-agent" in result.output


def test_validate_reports_bad_suite(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\ncases: [{id: a}]\n")
    result = invoke("validate", bad)
    assert result.exit_code == 2
    assert "input" in result.output


def test_run_stores_results_and_writes_summary(tmp_path):
    summary = tmp_path / "summary.md"
    result = invoke(
        "run", SUITES / "weather-agent.yaml", "--store", tmp_path / "s", "--summary-file", summary
    )
    assert result.exit_code == 0, result.output
    assert list((tmp_path / "s" / "runs").glob("weather-agent--fake-strong--*.json"))
    assert "`weather-agent` on `fake-strong`" in summary.read_text()


def test_run_fail_under(tmp_path):
    result = invoke("run", SUITES, "-m", "fake-sloppy", "--store", tmp_path, "--fail-under", "0.9")
    assert result.exit_code == 1


def test_baseline_then_gate_passes_and_catches_regression(tmp_path):
    base = tmp_path / "baseline.json"
    store = tmp_path / "store"
    assert invoke("baseline", SUITES, "--out", base).exit_code == 0
    assert set(json.loads(base.read_text())["suites"]) == {
        "code-search",
        "recipe-scaler",
        "trip-planner",
        "weather-agent",
    }

    ok = invoke("gate", SUITES, "--baseline", base, "--store", store)
    assert ok.exit_code == 0, ok.output
    assert "Eval gate passed" in ok.output

    out = tmp_path / "gate.json"
    regressed = invoke(
        "gate", SUITES, "-m", "fake-fast", "--baseline-model", "fake-strong",
        "--baseline", base, "--store", store, "--json-out", out,
    )  # fmt: skip
    assert regressed.exit_code == 1
    assert "Eval gate FAILED" in regressed.output
    data = json.loads(out.read_text())
    failed = {s["suite"] for s in data["suites"] if s["status"] == "fail"}
    assert failed == {"weather-agent", "recipe-scaler"}


def test_compare_and_leaderboard(tmp_path):
    result = invoke(
        "compare", SUITES / "weather-agent.yaml", "-a", "fake-strong", "-b", "fake-sloppy",
        "--store", tmp_path,
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    board = invoke("leaderboard", "--store", tmp_path, "--markdown")
    assert board.exit_code == 0
    first_row = next(line for line in board.output.splitlines() if line.startswith("| 1 |"))
    assert "fake-strong" in first_row


def test_runs_and_show(tmp_path):
    invoke("run", SUITES / "recipe-scaler.yaml", "--store", tmp_path)
    run_file = next((tmp_path / "runs").glob("*.json"))
    listed = invoke("runs", "--store", tmp_path)
    assert listed.exit_code == 0
    shown = invoke("show", run_file.stem, "--store", tmp_path)
    assert shown.exit_code == 0
    assert "recipe-scaler" in shown.output
    assert invoke("show", "nope", "--store", tmp_path).exit_code == 2


def test_version():
    result = invoke("--version")
    assert result.exit_code == 0
    assert "agent-evals-arena" in result.output
