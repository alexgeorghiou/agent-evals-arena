"""Smoke-test the Streamlit dashboard against a populated store (skipped without streamlit)."""

import sys
from pathlib import Path

import pytest

from evals_arena.arena import compare_runs
from evals_arena.elo import Leaderboard
from evals_arena.judge import HeuristicJudge, HeuristicPairwiseJudge
from evals_arena.loader import load_suite
from evals_arena.providers import create_provider
from evals_arena.runner import Runner
from evals_arena.storage import ResultStore

from .conftest import ROOT, SUITES

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = ROOT / "dashboard" / "app.py"


def seed(store_dir: Path) -> None:
    store = ResultStore(store_dir)
    suite = load_suite(SUITES / "weather-agent.yaml")
    runs = []
    for model in ("fake-strong", "fake-sloppy"):
        provider = create_provider(f"fake:{model}", fake_script_paths=[SUITES])
        run = Runner(provider, judge=HeuristicJudge()).run(suite)
        store.save(run)
        runs.append(run)
    compare_runs(suite, *runs, HeuristicPairwiseJudge(), Leaderboard(store.db_path))


def run_app(store_dir: Path) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=60)
    argv = sys.argv
    sys.argv = ["app.py", "--store", str(store_dir)]
    try:
        at.run()
    finally:
        sys.argv = argv
    return at


def test_dashboard_renders_with_data(tmp_path):
    seed(tmp_path)
    at = run_app(tmp_path)
    assert not at.exception, at.exception
    assert at.title[0].value == "agent-evals-arena"
    assert len(at.tabs) == 4
    assert any("2 runs" in c.value for c in at.caption)


def test_dashboard_handles_empty_store(tmp_path):
    at = run_app(tmp_path)
    assert not at.exception
    assert "No runs found" in at.warning[0].value
