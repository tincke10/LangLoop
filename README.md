# langgraph-generator-discriminator

[![CI](https://github.com/tincke10/LangLoop/actions/workflows/ci.yml/badge.svg)](https://github.com/tincke10/LangLoop/actions/workflows/ci.yml)

A **LangGraph** reimplementation of the **Generator-Discriminator (Ralph Loop)** pattern:
an iterative quality-improvement loop where a generator produces content and a
discriminator evaluates it, repeating until it reaches a quality threshold or an iteration
cap.

It's the "with a framework" version of a pattern that is originally written **by hand in
TypeScript** in the `Barto-MCP` repo. The point of this repo is to show **what LangGraph
gives you for free** that the hand-rolled version had to orchestrate itself.

---

## The idea in one picture

```
                    ┌──────────────────┐
        ┌──────────►│    GENERATOR     │
        │           │ (generate/improve)│
        │           └────────┬─────────┘
        │                    │  fixed edge: you generated -> you get evaluated
        │                    ▼
        │           ┌──────────────────┐
        │           │  DISCRIMINATOR   │
        │           │ (score+feedback) │
        │           └────────┬─────────┘
        │                    │
        │           should_continue(state)   ← CONDITIONAL edge (the router)
        │                    │
        │     ┌──────────────┴───────────────┐
        │     │                              │
   score < threshold                  score >= threshold
   and iters remain                       OR iter >= max
        │     │                              │
        └─────┘                              ▼
   (CYCLE: loop back                       [END]
    with the feedback)
```

The arrow that **loops back** from `discriminator` to `generator` is the **cycle**. That's
what a linear LangChain chain **cannot** model — and the reason LangGraph exists.

---

## Repo architecture

```
.
├── pyproject.toml                      # deps + config (uv)
├── env.example                         # variables template (copy to .env)
├── src/generator_discriminator/
│   ├── state.py                        # GraphState (Pydantic) + Evaluation (structured output)
│   ├── nodes.py                        # generator_node + discriminator_node
│   ├── graph.py                        # StateGraph: nodes, edges, cycle, MemorySaver
│   └── main.py                         # demo entrypoint
└── README.md
```

What each file solves:

| File        | What it solves                                                               |
|-------------|------------------------------------------------------------------------------|
| `state.py`  | The shared **State** (store) + the typed schema for the structured output.   |
| `nodes.py`  | The **work**: each node is a function `(state) -> dict` (a state patch).      |
| `graph.py`  | The **flow**: registers nodes, wires edges, defines the **cycle** and checkpointing. |
| `main.py`   | Loads `.env`, compiles the graph and runs it with `invoke`.                  |

---

## How to run it

```bash
# 1. Install deps (uv resolves Python 3.12 on its own)
uv sync --python 3.12

# 2. Configure the API key
cp env.example .env        # and fill in ANTHROPIC_API_KEY

# 3. Run the loop
uv run gd-loop
# or:  uv run python -m generator_discriminator.main
```

In the logs you'll see **every transition**: which node ran, on which iteration, with what
score, and the router's decision (continue or stop).

---

## Concepts

### 1. StateGraph
The central structure. A directed graph where you declare **nodes** (work) and **edges**
(flow). You give it a **state schema** (here a Pydantic model) and it knows how to
hydrate/validate that state at each step. You **compile** it (`.compile()`) into a runnable
app with `.invoke()` / `.stream()`.

### 2. State (shared state)
Nodes **don't pass data as arguments**. There is **one** state object that flows through
the whole graph. Each node **reads** it and returns a **patch** (a dict) with the fields to
update; LangGraph merges it. Mindset is **Redux**: State = store, node = reducer. By default
the merge **overwrites**; to **accumulate** (e.g. a list) you use a *reducer*
(`Annotated[list, add]`) or do it by hand reading+appending (which is what we do with
`history`).

### 3. Nodes
A function `(state) -> dict`. That's it. The **unit of work**. It returns only the fields it
changes, not the whole State.

### 4. Edges
The **control flow**, declarative and separated from the logic:
- **Fixed** (`add_edge`): "after A, always B".
- **Conditional** (`add_conditional_edges`): runs a function `(state) -> label` and jumps
  based on the result. **This is what creates the cycle.**

### 5. Cycle vs linear chain
A LangChain chain is a **DAG (acyclic)**: `A → B → C`, no going back. LangGraph allows
**cycles**: a conditional edge can send you back to a previous node. That's why it models
refinement loops, ReAct agents, human-in-the-loop, etc.

### 6. Iteration guard
The `max_iterations` stop condition lives in the conditional edge's function
(`should_continue`). Without it, a draft that never beats the threshold would create an
**infinite loop**. The guard is **mandatory** in any cyclic graph.

### 7. Checkpointing
`MemorySaver` stores a **snapshot of the State after each node**, indexed by `thread_id`. It
enables persistence, resuming a run, inspecting the history, time-travel and
human-in-the-loop. In production you'd use `SqliteSaver` / `PostgresSaver`.

### 8. Structured output
`llm.with_structured_output(Evaluation)` forces the model to return a valid Pydantic object
(here `{score, feedback}`) using the provider's tool-calling. No more parsing strings by hand.

---

## Barto-MCP (by hand) vs LangGraph (this repo)

| In Barto-MCP you did it like this (by hand)                | In LangGraph it's modeled like this                         |
|------------------------------------------------------------|-------------------------------------------------------------|
| An object you passed and mutated between functions         | **State** (Pydantic), merged by the framework               |
| `generator()` and `discriminator()` as loose functions     | **Nodes** registered with `add_node`                        |
| `while (...) { ... if (cond) break; }`                     | **Conditional edge** `add_conditional_edges` + `should_continue` |
| The `if (score >= th || iter >= max) break;` inside the while | The **router function**, declarative and separated       |
| Parsing the discriminator's text by hand                   | `with_structured_output(Evaluation)`, typed and guaranteed  |
| (you didn't have) state persistence/resume                 | **Checkpointing** with `MemorySaver`                        |
| Scattered `console.log`                                    | Per-transition logging in every node + the router           |




**1. Why LangGraph and not a LangChain chain for this problem?**
Because a chain is an **acyclic** graph (a DAG): it goes from A to B to C with no going
back. This problem needs a **cycle** — generate, evaluate, and if it doesn't reach the
threshold, **go back** and generate with the feedback. LangGraph supports conditional edges
that close that cycle; a chain doesn't. You see it in `graph.py`: the conditional edge maps
`"generator" -> "generator"`, and that loop-back arrow is the loop.

**2. How do nodes communicate if they don't pass arguments?**
Through the **shared State**. Each node receives the whole State, does its work, and returns
a **patch dict** with the fields it updates; LangGraph merges it into the State. It's the
Redux model: the State is the store, each node a reducer. The `generator` writes `draft`;
the `discriminator` reads that `draft` from the State and writes `score`/`feedback`.

**3. How do you avoid an infinite loop?**
With a **guard** in the conditional edge's function (`should_continue`). It has two stops:
`score >= threshold` (we achieved quality) **or** `iteration >= max_iterations` (we ran out
of attempts). The `generator` increments `iteration` on every pass, so the second stop is
always eventually met. Without that guard, a draft that never beats the threshold would run
forever.

**4. What's the checkpointer (`MemorySaver`) for if the graph already runs by itself?**
It persists a **snapshot of the State after each node**, indexed by `thread_id`. That
enables: resuming an interrupted run, inspecting the history step by step, *time-travel*
(going back to a previous state) and *human-in-the-loop* (pausing, letting a human edit the
State, and continuing). `MemorySaver` is in-memory for demo/tests; in production you'd use
`SqliteSaver` or `PostgresSaver`. That's why `invoke` requires a `thread_id` in `config`
when there's a checkpointer.

**5. How do you guarantee the discriminator returns a numeric score and not free text?**
With `llm.with_structured_output(Evaluation)`, where `Evaluation` is a Pydantic model with
`score: int (0-100)` and `feedback: str`. Under the hood it uses the provider's
**tool-calling** to force the model to produce an object matching the schema; Pydantic
validates types and ranges. If the model tried to return something invalid, validation fails
instead of propagating corrupt data. That's the difference from parsing the string by hand,
which is fragile.