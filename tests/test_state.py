"""Tests for the typed State and the structured-output schema.

What we prove here: Pydantic actually guards the data. A discriminator that tries to
return a score of 150 fails LOUDLY instead of poisoning the loop. This is the payoff of
using Pydantic over a plain dict/TypedDict.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from generator_discriminator.state import Evaluation, GraphState


def test_graphstate_applies_defaults():
    state = GraphState(task="write a paragraph about X")
    assert state.draft == ""
    assert state.score == 0
    assert state.iteration == 0
    assert state.max_iterations == 5
    assert state.threshold == 85
    assert state.history == []


def test_graphstate_requires_task():
    with pytest.raises(ValidationError):
        GraphState()  # task has no default -> required


def test_evaluation_accepts_valid_score():
    evaluation = Evaluation(score=85, feedback="good enough")
    assert evaluation.score == 85
    assert evaluation.feedback == "good enough"


def test_evaluation_rejects_score_above_100():
    with pytest.raises(ValidationError):
        Evaluation(score=150, feedback="impossible")


def test_evaluation_rejects_negative_score():
    with pytest.raises(ValidationError):
        Evaluation(score=-1, feedback="impossible")
