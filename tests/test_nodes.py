"""Tests for the nodes, with the LLM mocked.

We assert on the CONTRACT of each node (the patch it returns and the prompt it builds),
never on the model's actual text. The fake gives us deterministic inputs.
"""

from __future__ import annotations

from fakes import FakeLLM

from generator_discriminator.nodes import discriminator_node, generator_node
from generator_discriminator.state import Evaluation, GraphState


def _human_content(fake: FakeLLM, call_index: int = 0) -> str:
    """The text of the HumanMessage from the call_index-th generator invocation."""
    messages = fake.invoke_messages[call_index]
    return messages[-1].content


def test_generator_increments_iteration_and_sets_draft(patch_llm):
    patch_llm(FakeLLM(draft="hello world"))
    patch = generator_node(GraphState(task="write X"))

    assert patch["draft"] == "hello world"
    assert patch["iteration"] == 1
    # The generator must NOT touch score/feedback — that's the discriminator's job.
    assert "score" not in patch
    assert "feedback" not in patch


def test_generator_uses_generate_mode_on_first_pass(patch_llm):
    fake = patch_llm(FakeLLM())
    generator_node(GraphState(task="write X"))  # no feedback yet

    assert "feedback" not in _human_content(fake).lower()


def test_generator_uses_improve_mode_when_feedback_present(patch_llm):
    fake = patch_llm(FakeLLM())
    state = GraphState(task="write X", draft="old draft", score=40, feedback="add examples", iteration=1)
    patch = generator_node(state)

    prompt = _human_content(fake)
    assert "feedback" in prompt.lower()
    assert "add examples" in prompt  # the actual feedback is fed back in
    assert patch["iteration"] == 2  # incremented from 1


def test_discriminator_returns_structured_score_and_feedback(patch_llm):
    patch_llm(FakeLLM(evaluations=[Evaluation(score=73, feedback="needs work")]))
    patch = discriminator_node(GraphState(task="x", draft="some draft", iteration=1))

    assert patch["score"] == 73
    assert patch["feedback"] == "needs work"


def test_discriminator_appends_to_history(patch_llm):
    patch_llm(FakeLLM(evaluations=[Evaluation(score=73, feedback="fb")]))
    state = GraphState(
        task="x",
        draft="d",
        iteration=2,
        history=[{"iteration": 1, "score": 50, "feedback": "prev"}],
    )
    patch = discriminator_node(state)

    assert len(patch["history"]) == 2  # previous entry + the new one
    assert patch["history"][-1] == {"iteration": 2, "score": 73, "feedback": "fb"}
