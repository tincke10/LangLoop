"""Shared pytest fixtures.

The key fixture is `patch_llm`: it swaps the real `_make_llm()` in the nodes module for a
FakeLLM. Because the nodes call `_make_llm()` by name at runtime, monkeypatching the module
attribute is enough — no need to touch ChatAnthropic directly.
"""

from __future__ import annotations

import pytest
from fakes import FakeLLM


@pytest.fixture
def patch_llm(monkeypatch):
    """Return a helper that installs a FakeLLM and hands it back for assertions.

    Usage:
        fake = patch_llm(FakeLLM(evaluations=[...]))

    We patch `_make_llm` to return THE SAME fake on every call, so its internal counters
    persist across all generator/discriminator invocations within one graph run.
    """

    def _install(fake: FakeLLM) -> FakeLLM:
        monkeypatch.setattr("generator_discriminator.nodes._make_llm", lambda: fake)
        return fake

    return _install
