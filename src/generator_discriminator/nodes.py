"""The graph nodes: generator and discriminator.

KEY CONCEPT (#2 of the mental model):
A node is JUST a function `(state) -> dict`. It receives the whole State, does its work,
and returns a dictionary with the fields it wants to update. LangGraph merges that dict
into the State. You don't return the whole State: you return the PATCH.
"""

from __future__ import annotations

import logging
import os

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from .state import Evaluation, GraphState

logger = logging.getLogger(__name__)


def _make_llm():
    """Build the chat model.

    Why a function and not a module-level constant: so we read the env vars AFTER main.py has
    run load_dotenv(). If we instantiated it at import time, the .env wouldn't be loaded yet.

    init_chat_model is the provider-agnostic factory: it takes a model name string plus the
    model_provider and returns the right ChatModel instance. This lets the model, provider,
    API key, and base URL all be swapped via env vars (see env.example).
    """
    model = os.getenv("MODEL", "claude-sonnet-4-6")
    model_provider = os.getenv("MODEL_PROVIDER", "anthropic")
    api_key = os.getenv("API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    base_url = os.getenv("BASE_URL")

    # A medium-high temperature in the generator favors variety across iterations;
    # here we use a middle value that works for both nodes.
    kwargs: dict = {
        "model": model,
        "model_provider": model_provider,
        "temperature": 0.7,
        "max_tokens": 1024,
    }
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url

    # Thinking mode is incompatible with the forced tool_choice used by the discriminator's
    # structured output, so we disable it for the anthropic provider.
    if model_provider == "anthropic":
        kwargs["thinking"] = {"type": "disabled"}

    return init_chat_model(**kwargs)


# ---------------------------------------------------------------------------
# NODE 1: GENERATOR
# ---------------------------------------------------------------------------
def generator_node(state: GraphState) -> dict:
    """Generate (or IMPROVE) the draft.

    - Iteration 1 (no feedback yet): generate from scratch from the task.
    - Following iterations: take the discriminator's feedback and IMPROVE the draft.

    This "take feedback and generate again" is exactly the left-hand side of your Ralph
    Loop in Barto-MCP. The difference: here there's no `while`, the graph orchestrates it
    via the conditional edge (see graph.py).
    """
    llm = _make_llm()
    next_iteration = state.iteration + 1

    system = SystemMessage(
        content=(
            "You are an expert writer. Produce clear, precise, well-structured content. "
            "Return ONLY the requested content, with no preamble or meta-commentary."
        )
    )

    if state.feedback:
        # There is feedback from a previous round -> IMPROVE mode.
        human = HumanMessage(
            content=(
                f"Task: {state.task}\n\n"
                f"Your previous draft was:\n{state.draft}\n\n"
                f"A reviewer scored it {state.score}/100 with this feedback:\n{state.feedback}\n\n"
                "Rewrite the content applying the feedback to raise the quality."
            )
        )
        mode = "improve"
    else:
        # First pass -> generate from scratch.
        human = HumanMessage(content=f"Task: {state.task}\n\nWrite the requested content.")
        mode = "generate"

    response = llm.invoke([system, human])
    draft = response.content if isinstance(response.content, str) else str(response.content)

    # Log the transition (spec requirement): which node ran + on which iteration.
    logger.info("[generator] mode=%s iteration=%d -> draft of %d chars", mode, next_iteration, len(draft))

    # Return the PATCH. We don't touch score/feedback here: that's the discriminator's job.
    return {
        "draft": draft,
        "iteration": next_iteration,
    }


# ---------------------------------------------------------------------------
# NODE 2: DISCRIMINATOR
# ---------------------------------------------------------------------------
def discriminator_node(state: GraphState) -> dict:
    """Evaluate the draft and return a STRUCTURED score + feedback.

    Here's the hard requirement: with_structured_output(Evaluation). The LLM is forced to
    return a valid Evaluation object (score 0-100 + feedback), not a loose string. In
    Barto-MCP you parsed this by hand; here the framework guarantees it.
    """
    llm = _make_llm()
    # .with_structured_output() wraps the model: instead of text, it returns an instance
    # of Evaluation (it uses Anthropic's tool-calling under the hood).
    evaluator = llm.with_structured_output(Evaluation)

    system = SystemMessage(
        content=(
            "You are a demanding but fair quality reviewer. You score content from 0 to 100 "
            "based on clarity, accuracy, structure, and how well it fulfills the requested task. "
            "Be concrete in your feedback: say WHAT to improve, not generalities."
        )
    )
    human = HumanMessage(
        content=(
            f"Original task: {state.task}\n\n"
            f"Content to evaluate:\n{state.draft}\n\n"
            "Score it from 0 to 100 and give actionable feedback."
        )
    )

    evaluation: Evaluation = evaluator.invoke([system, human])

    logger.info(
        "[discriminator] iteration=%d -> score=%d (threshold=%d)",
        state.iteration,
        evaluation.score,
        state.threshold,
    )

    # Accumulate this round's snapshot into history (read + append + return new list).
    new_history = state.history + [
        {
            "iteration": state.iteration,
            "score": evaluation.score,
            "feedback": evaluation.feedback,
        }
    ]

    return {
        "score": evaluation.score,
        "feedback": evaluation.feedback,
        "history": new_history,
    }
