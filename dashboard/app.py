"""Streamlit dashboard for agent-evals-arena.

Run with:  streamlit run dashboard/app.py -- --store .arena
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from evals_arena.elo import Leaderboard
from evals_arena.results import RunResult
from evals_arena.storage import ResultStore


def _store_path() -> Path:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", default=".arena")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return Path(args.store)


@st.cache_data(ttl=10)
def load_index(root: str) -> pd.DataFrame:
    rows = ResultStore(root).list(limit=500)
    return pd.DataFrame(rows)


@st.cache_data(ttl=10)
def load_run(root: str, run_id: str) -> dict:
    return ResultStore(root).load(run_id).model_dump()


def runs_tab(index: pd.DataFrame) -> None:
    st.subheader("Runs")
    suites = sorted(index["suite"].unique())
    models = sorted(index["model"].unique())
    col1, col2 = st.columns(2)
    chosen_suites = col1.multiselect("Suite", suites, default=suites)
    chosen_models = col2.multiselect("Model", models, default=models)
    view = index[index["suite"].isin(chosen_suites) & index["model"].isin(chosen_models)]
    st.dataframe(
        view[
            [
                "run_id",
                "suite",
                "model",
                "score",
                "pass_rate",
                "cost_usd",
                "latency_p95",
                "started_at",
            ]
        ],
        hide_index=True,
        use_container_width=True,
        column_config={
            "score": st.column_config.ProgressColumn("score", min_value=0, max_value=1),
            "pass_rate": st.column_config.NumberColumn("pass rate", format="%.0f%%"),
            "cost_usd": st.column_config.NumberColumn("cost $", format="%.5f"),
        },
    )
    if not view.empty:
        st.markdown("**Score over time**")
        trend = view.sort_values("started_at").pivot_table(
            index="started_at", columns="model", values="score", aggfunc="mean"
        )
        st.line_chart(trend)


def drilldown_tab(root: str, index: pd.DataFrame) -> None:
    st.subheader("Case drilldown")
    run_id = st.selectbox("Run", index["run_id"].tolist())
    if not run_id:
        return
    run = RunResult.model_validate(load_run(root, run_id))
    assert run.summary is not None
    s = run.summary
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Score", f"{s.score:.3f}", help=f"95% CI {s.score_ci_low:.2f}-{s.score_ci_high:.2f}")
    m2.metric("Pass rate", f"{s.pass_rate:.0%}", help=f"{s.passed}/{s.cases} cases")
    m3.metric("p95 latency", f"{s.latency_p95_ms:.0f} ms")
    m4.metric("Cost", f"${s.total_cost_usd:.4f}")

    only_failed = st.toggle("Only failing cases", value=False)
    for case in run.cases:
        if only_failed and case.passed:
            continue
        label = f"{'PASS' if case.passed else 'FAIL'}  {case.case_id}  -  score {case.score:.2f}"
        with st.expander(label, expanded=not case.passed):
            st.markdown(f"**Input**\n\n{case.input}")
            st.markdown("**Output**")
            st.code(case.output or "<empty>", language="markdown")
            if case.trajectory:
                st.markdown("**Trajectory**")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "turn": t["turn"],
                                "tool": t["name"],
                                "arguments": json.dumps(t["arguments"]),
                                "result": json.dumps(t["result"])[:200] if t["result"] else "",
                                "error": t["error"] or "",
                            }
                            for t in case.trajectory
                        ]
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
            st.markdown("**Assertions**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "assertion": a.name,
                            "type": a.type,
                            "passed": a.passed,
                            "score": a.score,
                            "weight": a.weight,
                            "message": a.message,
                        }
                        for a in case.assertions
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                f"{case.turns} turns - {case.input_tokens} in / {case.output_tokens} out tokens - "
                f"{case.latency_ms:.0f} ms - ${case.cost_usd:.5f}"
            )


def leaderboard_tab(root: str) -> None:
    st.subheader("Elo leaderboard")
    board = Leaderboard(ResultStore(root).db_path)
    standings = board.standings()
    if not standings:
        st.info("No pairwise matches yet. Run `arena compare suites -a <model> -b <model>`.")
        return
    df = pd.DataFrame(
        [
            {
                "model": s.model,
                "elo": round(s.rating),
                "games": s.games,
                "wins": s.wins,
                "losses": s.losses,
                "ties": s.ties,
                "win rate": s.win_rate,
            }
            for s in standings
        ]
    )
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={"win rate": st.column_config.ProgressColumn(min_value=0, max_value=1)},
    )
    st.bar_chart(df.set_index("model")["elo"])
    with st.expander("Recent matches"):
        st.dataframe(pd.DataFrame(board.matches(200)), hide_index=True)


def cost_tab(index: pd.DataFrame) -> None:
    st.subheader("Cost vs quality")
    latest = index.sort_values("started_at").groupby(["suite", "model"]).tail(1)
    st.scatter_chart(latest, x="cost_usd", y="score", color="model", size="cases")
    st.markdown("**Latency (p50 / p95, ms) by model**")
    lat = latest.groupby("model")[["latency_p50", "latency_p95"]].mean()
    st.bar_chart(lat)
    st.markdown("**Total spend by model**")
    st.bar_chart(index.groupby("model")["cost_usd"].sum())


def main() -> None:
    st.set_page_config(page_title="agent-evals-arena", layout="wide")
    st.title("agent-evals-arena")
    root = str(_store_path())
    index = load_index(root)
    st.caption(f"store: `{root}` - {len(index)} runs")
    if index.empty:
        st.warning("No runs found. Try `arena run suites` first.")
        return
    tabs = st.tabs(["Runs", "Drilldown", "Leaderboard", "Cost & latency"])
    with tabs[0]:
        runs_tab(index)
    with tabs[1]:
        drilldown_tab(root, index)
    with tabs[2]:
        leaderboard_tab(root)
    with tabs[3]:
        cost_tab(index)


main()
