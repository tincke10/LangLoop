"""End-to-end tests of the compiled graph, with the LLM mocked.

These are the headline tests: they run the WHOLE cycle (generator -> discriminator ->
router -> ...) and prove the loop behaves. Each test creates its own graph, so each gets a
fresh MemorySaver and the runs don't bleed into each other.

The trick: a FakeLLM seeded with a sequence of scores drives the loop deterministically.
"""

from __future__ import annotations

from fakes import FakeLLM

from generator_discriminator.graph import build_graph
from generator_discriminator.state import Evaluation

CONFIG = {"configurable": {"thread_id": "test"}}


def test_single_pass_when_first_draft_already_good(patch_llm):
    # First evaluation already clears the threshold -> the loop stops after one pass.
    patch_llm(FakeLLM(evaluations=[Evaluation(score=95, feedback="great")]))
    app = build_graph()

    final = app.invoke({"task": "x", "threshold": 85, "max_iterations": 5}, config=CONFIG)

    assert final["iteration"] == 1
    assert final["score"] == 95
    assert len(final["history"]) == 1


def test_loop_converges_when_threshold_reached(patch_llm):
    # Rising scores: 50 -> 70 -> 90. It should iterate three times and stop on the third.
    patch_llm(
        FakeLLM(
            evaluations=[
                Evaluation(score=50, feedback="f1"),
                Evaluation(score=70, feedback="f2"),
                Evaluation(score=90, feedback="f3"),
            ]
        )
    )
    app = build_graph()

    final = app.invoke({"task": "x", "threshold": 85, "max_iterations": 5}, config=CONFIG)

    assert final["iteration"] == 3
    assert final["score"] == 90
    # History captured every round, in order.
    assert [h["score"] for h in final["history"]] == [50, 70, 90]


def test_guard_stops_loop_when_threshold_never_reached(patch_llm):
    # Score is always 10, never beats 85 -> the max_iterations guard must stop it.
    patch_llm(FakeLLM(evaluations=[Evaluation(score=10, feedback="bad")]))
    app = build_graph()

    final = app.invoke({"task": "x", "threshold": 85, "max_iterations": 3}, config=CONFIG)

    assert final["iteration"] == 3  # stopped exactly at the cap, not 4, not infinite
    assert final["score"] == 10
    assert len(final["history"]) == 3
