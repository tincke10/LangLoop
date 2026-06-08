"""Test doubles for the LLM. No network, fully deterministic.

This is the heart of the testing strategy: we DON'T test Anthropic (not our code,
non-deterministic, costs money). We test OUR graph logic by swapping ChatAnthropic for
these fakes via monkeypatch (see conftest.py).

The fake must satisfy the two ways the nodes use the model:
  - generator_node:     llm.invoke([...]).content              -> a string draft
  - discriminator_node: llm.with_structured_output(Evaluation)
                            .invoke([...])                      -> an Evaluation instance
"""

from __future__ import annotations

from generator_discriminator.state import Evaluation


class FakeResponse:
    """Mimics the message object returned by llm.invoke(): only needs `.content`."""

    def __init__(self, content: str):
        self.content = content


class FakeStructuredLLM:
    """Mimics llm.with_structured_output(Evaluation).

    Returns a pre-seeded sequence of Evaluation objects, one per call. Once the sequence
    is exhausted it keeps returning the last one. This lets a test drive the loop with,
    e.g., rising scores [50, 70, 90] to prove the cycle converges.
    """

    def __init__(self, evaluations: list[Evaluation]):
        self._evaluations = list(evaluations)
        self.calls = 0  # how many times the discriminator asked for an evaluation

    def invoke(self, messages):
        idx = min(self.calls, len(self._evaluations) - 1)
        self.calls += 1
        return self._evaluations[idx]


class FakeLLM:
    """Drop-in replacement for ChatAnthropic in tests.

    A single instance is reused across the whole graph run (see the patch_llm fixture),
    so its internal call counter survives across iterations — that's what makes the
    end-to-end loop tests deterministic.
    """

    def __init__(self, draft: str = "a draft", evaluations: list[Evaluation] | None = None):
        self.draft = draft
        self.evaluations = evaluations or [Evaluation(score=100, feedback="perfect")]
        # Record every set of messages the generator sent, so tests can assert on the
        # prompt (e.g. that "improve" mode includes the feedback).
        self.invoke_messages: list = []
        self._structured = FakeStructuredLLM(self.evaluations)

    # Used by generator_node.
    def invoke(self, messages):
        self.invoke_messages.append(messages)
        return FakeResponse(self.draft)

    # Used by discriminator_node.
    def with_structured_output(self, schema):
        assert schema is Evaluation, "the discriminator must request the Evaluation schema"
        return self._structured
