import pytest

from evals_arena.judge import (
    HeuristicJudge,
    HeuristicPairwiseJudge,
    JudgeParseError,
    LLMJudge,
    LLMPairwiseJudge,
    PairwiseJudge,
    extract_json,
    parse_preference,
    parse_rubric_reply,
    prefer_with_swap,
)
from evals_arena.providers import Completion, Provider
from evals_arena.schema import Rubric

RUBRIC = Rubric.model_validate(
    {
        "criteria": [
            {"name": "accuracy", "description": "correct", "weight": 3},
            {"name": "tone", "description": "polite", "weight": 1},
        ]
    }
)


class Canned(Provider):
    vendor = "canned"

    def __init__(self, *replies: str) -> None:
        super().__init__("canned-judge")
        self.replies = list(replies)
        self.prompts: list[str] = []

    def complete(self, messages, **kwargs) -> Completion:
        self.prompts.append(messages[-1].content)
        return Completion(text=self.replies.pop(0))


def test_extract_json_handles_fences_and_prose():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Result: {"a": {"b": "x}"}} trailing') == {"a": {"b": "x}"}}
    assert extract_json('{not json} then {"ok": true}') == {"ok": True}
    with pytest.raises(JudgeParseError):
        extract_json("no braces here")


def test_parse_rubric_reply_clamps_and_fills_missing():
    reply = '{"criteria": [{"name": "Accuracy", "score": 9, "reason": "great"}], "rationale": "ok"}'
    scores, rationale = parse_rubric_reply(reply, RUBRIC)
    assert [(s.name, s.score) for s in scores] == [("accuracy", 5), ("tone", 1)]
    assert scores[1].reason == "not scored"
    assert rationale == "ok"


def test_parse_rubric_reply_accepts_mapping_form():
    scores, _ = parse_rubric_reply('{"scores": {"accuracy": 4, "tone": 2}}', RUBRIC)
    assert [s.score for s in scores] == [4, 2]


def test_parse_rubric_reply_rejects_non_numeric():
    with pytest.raises(JudgeParseError):
        parse_rubric_reply('{"criteria": [{"name": "accuracy", "score": "high"}]}', RUBRIC)


def test_llm_judge_weights_scores():
    provider = Canned(
        '{"criteria": [{"name": "accuracy", "score": 5}, {"name": "tone", "score": 1}]}'
    )
    verdict = LLMJudge(provider).grade(RUBRIC, task="2+2?", output="4", reference="4")
    # accuracy: 1.0 * 3, tone: 0.0 * 1 -> 0.75
    assert verdict.normalized == pytest.approx(0.75)
    assert "REFERENCE ANSWER" in provider.prompts[0]
    assert "accuracy (weight 3)" in provider.prompts[0]


@pytest.mark.parametrize(
    ("reply", "expected"),
    [('{"winner": "a"}', "A"), ('{"winner": "B"}', "B"), ('{"winner": "tie"}', "tie")],
)
def test_parse_preference(reply, expected):
    assert parse_preference(reply) == expected


def test_parse_preference_rejects_garbage():
    with pytest.raises(JudgeParseError):
        parse_preference('{"winner": "both"}')


class Biased(PairwiseJudge):
    """Always prefers whatever is shown first - the classic position bias."""

    def prefer(self, **kwargs):
        return "A"


def test_position_swap_neutralises_bias():
    assert prefer_with_swap(Biased(), task="t", output_a="x", output_b="y") == "tie"


def test_position_swap_keeps_consistent_verdicts():
    judge = LLMPairwiseJudge(Canned('{"winner": "A"}', '{"winner": "B"}'))
    assert prefer_with_swap(judge, task="t", output_a="good", output_b="bad") == "A"


def test_heuristic_judge_keywords_and_reference():
    rubric = Rubric.model_validate(
        {
            "criteria": [
                {"name": "kw", "description": "", "keywords": ["paris", "sunny"]},
                {"name": "ref", "description": ""},
            ]
        }
    )
    verdict = HeuristicJudge().grade(
        rubric, task="t", output="Paris is cloudy", reference="Paris cloudy cold"
    )
    kw, ref = verdict.scores
    assert kw.score == 3.0  # 1/2 keywords on a 1..5 scale
    assert ref.score == pytest.approx(1 + 4 * 2 / 3, abs=0.01)
    assert 0 < verdict.normalized < 1


def test_heuristic_pairwise_is_symmetric():
    judge = HeuristicPairwiseJudge()
    ref = "Lisbon is warmer at 22 C"
    assert (
        judge.prefer(task="t", output_a="Lisbon 22 C warmer", output_b="no idea", reference=ref)
        == "A"
    )
    assert (
        judge.prefer(task="t", output_a="no idea", output_b="Lisbon 22 C warmer", reference=ref)
        == "B"
    )
    assert judge.prefer(task="t", output_a="same", output_b="same", reference=ref) == "tie"
