from pathlib import Path

import pytest
from pydantic import ValidationError

from evals_arena.loader import SuiteLoadError, discover, load_suite, load_suites
from evals_arena.schema import JudgeAssertion, ToolSequenceAssertion

from .conftest import make_suite


def test_assertions_are_discriminated_by_type():
    s = make_suite(
        cases=[
            {
                "id": "c1",
                "input": "x",
                "assertions": [
                    {"type": "tool_sequence", "steps": ["lookup", {"tool": "noop"}]},
                    {"type": "judge", "rubric": "basic"},
                ],
            }
        ]
    )
    seq, judge = s.cases[0].assertions
    assert isinstance(seq, ToolSequenceAssertion)
    assert [st.tool for st in seq.normalised_steps()] == ["lookup", "noop"]
    assert isinstance(judge, JudgeAssertion)


def test_unknown_assertion_type_is_rejected():
    with pytest.raises(ValidationError, match="does_not_exist"):
        make_suite(cases=[{"id": "c1", "input": "x", "assertions": [{"type": "does_not_exist"}]}])


def test_unknown_rubric_and_tool_references_are_rejected():
    with pytest.raises(ValidationError, match="unknown rubric"):
        make_suite(
            cases=[{"id": "c1", "input": "x", "assertions": [{"type": "judge", "rubric": "nope"}]}]
        )
    with pytest.raises(ValidationError, match="unknown tool"):
        make_suite(cases=[{"id": "c1", "input": "x", "tools": ["missing"]}])


def test_duplicate_case_ids_rejected():
    with pytest.raises(ValidationError, match="duplicate case ids: a"):
        make_suite(cases=[{"id": "a", "input": "x"}, {"id": "a", "input": "y"}])


def test_extra_keys_are_forbidden():
    with pytest.raises(ValidationError):
        make_suite(cases=[{"id": "c1", "input": "x", "inptu": "typo"}])


def test_tools_for_case_subset(suite):
    case = suite.cases[0].model_copy(update={"tools": ["noop"]})
    assert [t.name for t in suite.tools_for(case)] == ["noop"]
    assert len(suite.tools_for(suite.cases[0])) == 2


def test_load_suite_reports_location(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: Bad Name\ncases: []\n")
    with pytest.raises(SuiteLoadError) as err:
        load_suite(bad)
    assert "name" in err.value.message
    assert "cases" in err.value.message


def test_load_suite_rejects_non_mapping(tmp_path: Path):
    f = tmp_path / "list.yaml"
    f.write_text("- 1\n- 2\n")
    with pytest.raises(SuiteLoadError, match="mapping"):
        load_suite(f)


def test_bundled_suites_validate(suites_dir: Path):
    loaded = load_suites([suites_dir])
    assert loaded, "expected example suites"
    assert all(not p.name.endswith(".fake.yaml") for p in discover([suites_dir]))
