"""Tests for should_continue: the conditional edge's decision function.

This is THE most important unit to test. It's the brain of the cycle and it's pure logic
(no LLM, no network). It decides whether the loop keeps going or stops.
"""

from __future__ import annotations

from generator_discriminator.graph import should_continue
from generator_discriminator.state import GraphState


def _state(**overrides) -> GraphState:
    base = dict(task="x", score=0, threshold=85, iteration=0, max_iterations=5)
    base.update(overrides)
    return GraphState(**base)


def test_stops_when_score_meets_threshold_exactly():
    # Boundary: score == threshold must already stop (>=, not >).
    assert should_continue(_state(score=85, threshold=85, iteration=1)) == "end"


def test_stops_when_score_above_threshold():
    assert should_continue(_state(score=99, iteration=1)) == "end"


def test_guard_stops_on_max_iterations_even_with_low_score():
    # Low score but out of attempts -> the anti-infinite-loop guard fires.
    assert should_continue(_state(score=10, iteration=5, max_iterations=5)) == "end"


def test_continues_when_below_threshold_and_iterations_remain():
    assert should_continue(_state(score=40, iteration=2, max_iterations=5)) == "generator"


def test_threshold_wins_when_both_conditions_met():
    # Even if we also hit max_iterations, a good score still ends cleanly.
    assert should_continue(_state(score=90, iteration=5, max_iterations=5)) == "end"
