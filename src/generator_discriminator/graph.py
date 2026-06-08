"""StateGraph construction: nodes + edges + cycle + checkpointing.

KEY CONCEPTS (#3 and #4 of the mental model):
- Edges = the control flow. There are fixed ones (add_edge) and conditional ones (add_conditional_edges).
- The conditional edge is what creates the CYCLE: it loops back from discriminator to generator.
  That's what a linear LangChain chain CANNOT do. It's LangGraph's whole reason to exist.
"""

from __future__ import annotations

import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .nodes import discriminator_node, generator_node
from .state import GraphState

logger = logging.getLogger(__name__)


def should_continue(state: GraphState) -> str:
    """The DECISION function of the conditional edge. The brain of the loop.

    It looks at the State and returns a label ("end" or "generator"). LangGraph uses that
    label to know which node to jump to. This replaces, declaratively, the
    `if (score >= threshold || iteration >= max) break;` that in Barto-MCP lived buried
    inside a `while`.

    Two stop conditions (the second is the spec's anti-infinite-loop guard):
      1. score >= threshold  -> the draft is already good enough.
      2. iteration >= max_iterations -> we ran out of attempts; stop anyway.
    """
    if state.score >= state.threshold:
        logger.info("[router] score %d >= threshold %d -> END", state.score, state.threshold)
        return "end"

    if state.iteration >= state.max_iterations:
        logger.info(
            "[router] iteration %d >= max %d -> END (anti infinite-loop guard)",
            state.iteration,
            state.max_iterations,
        )
        return "end"

    logger.info("[router] score %d < threshold -> back to GENERATOR with the feedback", state.score)
    return "generator"


def build_graph():
    """Build and compile the graph. Returns a runnable app (.invoke / .stream)."""

    # 1) Create the graph declaring WHICH state schema it uses. This lets LangGraph know
    #    how to hydrate/validate the State at each node (it coerces it to GraphState).
    workflow = StateGraph(GraphState)

    # 2) Register the nodes. Logical name -> function. (spec's add_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("discriminator", discriminator_node)

    # 3) Entry point: where the graph starts. (spec's set_entry_point)
    workflow.set_entry_point("generator")

    # 4) FIXED edge: after the generator we ALWAYS go to the discriminator.
    #    There's no decision to make here: you generated -> you get evaluated.
    workflow.add_edge("generator", "discriminator")

    # 5) CONDITIONAL edge: the cycle. (spec's add_conditional_edges)
    #    After the discriminator we run should_continue(state). Depending on what it returns:
    #      "generator" -> generate again (CLOSES THE CYCLE)
    #      "end"       -> we're done
    #    The third argument is the "map": label -> actual destination.
    workflow.add_conditional_edges(
        "discriminator",
        should_continue,
        {
            "generator": "generator",  # <- this loop-back arrow is the CYCLE
            "end": END,                # END is LangGraph's special terminal node
        },
    )

    # 6) Checkpointing (spec requirement): MemorySaver stores a snapshot of the State
    #    after EVERY node, in memory, indexed by thread_id. This gives you:
    #      - state persistence between steps
    #      - the ability to resume / inspect the run history
    #      - the foundation for human-in-the-loop and time-travel
    #    (In production you'd use SqliteSaver/PostgresSaver; MemorySaver is for demo/tests.)
    checkpointer = MemorySaver()

    # 7) compile() freezes the topology and gives us the runnable app.
    return workflow.compile(checkpointer=checkpointer)
