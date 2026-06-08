"""Typed graph state + the discriminator's structured-output schema.

KEY CONCEPT (#1 of the mental model):
In LangGraph nodes do NOT pass data to each other as function arguments. There is ONE
shared state object that lives for the whole graph run. Each node reads it and returns a
"patch" (a dict) with the fields it wants to update; LangGraph merges that patch into the
state. Think Redux: the State is the store, each node is a reducer.

Why Pydantic instead of a TypedDict:
- Runtime validation (score 0-100 guaranteed, not just any int).
- Declarative defaults (iteration starts at 0 without anyone setting it).
- It's the SAME model we reuse for the discriminator's structured output.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Evaluation(BaseModel):
    """STRUCTURED output of the discriminator.

    This model is the contract we hand to `llm.with_structured_output(Evaluation)`.
    The provider (Claude) is FORCED to return exactly {score, feedback} with the right
    types. No more parsing a string by hand and praying the JSON comes back well-formed.

    In Barto-MCP you did this by hand: the discriminator returned text and you parsed it.
    Here the framework + Pydantic guarantee it typed.
    """

    score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Draft quality from 0 to 100. 100 = perfect, meets every criterion.",
    )
    feedback: str = Field(
        ...,
        description=(
            "Concrete, actionable critique: what is missing or what to improve to raise the score. "
            "If the score is already high, briefly explain why it's good."
        ),
    )


class GraphState(BaseModel):
    """The shared state that flows through the ENTIRE graph.

    Each field is a "channel" that nodes read from and write to. Notice how each one maps
    1:1 to a requirement of the spec.
    """

    # --- User input ---
    task: str = Field(..., description="The prompt. E.g. 'write a paragraph about X'.")

    # --- Produced by the generator ---
    draft: str = Field(default="", description="The content generated in the latest iteration.")

    # --- Produced by the discriminator (structured output) ---
    score: int = Field(default=0, description="Latest 0-100 score given by the discriminator.")
    feedback: str = Field(default="", description="Latest feedback. The generator uses it to improve.")

    # --- Loop control ---
    iteration: int = Field(default=0, description="How many times the generator ran. The guard uses it.")
    max_iterations: int = Field(default=5, description="Hard cap on iterations. Prevents infinite loops.")
    threshold: int = Field(default=85, description="Minimum score to stop and accept the draft as good.")

    # --- Traceability ---
    # We accumulate one snapshot per iteration so we can show how the loop evolved.
    # We manage it by hand in the nodes (read history, append, return the new list)
    # instead of using an Annotated reducer: more explicit and easier to defend.
    history: list[dict] = Field(
        default_factory=list,
        description="One record per iteration: {iteration, score, feedback}.",
    )
