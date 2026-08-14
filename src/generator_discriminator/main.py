"""Demo entrypoint. Loads the .env, builds the graph and runs the loop.

Run with:  uv run gd-loop
      or:  uv run python -m generator_discriminator.main
"""

from __future__ import annotations

import logging
import os
import sys

from dotenv import load_dotenv

from .graph import build_graph


def _configure_logging() -> None:
    """Basic logging of every transition (spec requirement)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    # 1) Load ANTHROPIC_API_KEY (and GD_MODEL if present) from .env BEFORE touching the LLM.
    load_dotenv()
    _configure_logging()

    # Accept either the provider-specific API_KEY or the legacy ANTHROPIC_API_KEY.
    if not (os.getenv("API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print(
            "ERROR: API key is missing. Copy env.example to .env and set API_KEY "
            "(or ANTHROPIC_API_KEY as fallback)."
        )
        sys.exit(1)

    # 2) Build the app (graph compiled with a checkpointer).
    app = build_graph()

    # 3) thread_id: identifies this run in the checkpointer. With the same thread_id you
    #    could resume/inspect the saved state. It's REQUIRED whenever the compiled graph
    #    has a checkpointer.
    config = {"configurable": {"thread_id": "demo-1"}}

    # 4) Initial state. We only set what the user provides; the rest uses the Pydantic
    #    model defaults (draft="", score=0, iteration=0, ...).
    initial_state = {
        "task": "Write a clear and engaging paragraph explaining what LangGraph is "
        "and why it's useful for cyclic LLM workflows.",
        "max_iterations": 5,
        "threshold": 85,
    }

    print("=" * 70)
    print("GENERATOR-DISCRIMINATOR LOOP (Ralph Loop in LangGraph)")
    print("=" * 70)
    print(f"Task: {initial_state['task']}\n")

    # 5) invoke() runs the graph until it reaches END and returns the final State (dict).
    final_state = app.invoke(initial_state, config=config)

    # 6) Run report.
    print("\n" + "=" * 70)
    print("LOOP EVOLUTION")
    print("=" * 70)
    for entry in final_state["history"]:
        print(f"  iteration {entry['iteration']}: score={entry['score']}")
        print(f"      feedback: {entry['feedback']}")

    print("\n" + "=" * 70)
    print(f"FINAL RESULT  (score={final_state['score']}, iterations={final_state['iteration']})")
    print("=" * 70)
    print(final_state["draft"])


if __name__ == "__main__":
    main()
