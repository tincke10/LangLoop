"""Los nodos del grafo: generator y discriminator.

CONCEPTO CLAVE (#2 del modelo mental):
Un nodo es SOLO una función `(state) -> dict`. Recibe el State completo, hace su
laburo, y devuelve un diccionario con los campos que quiere actualizar. LangGraph
mergea ese dict al State. No devolvés el State entero: devolvés el PARCHE.
"""

from __future__ import annotations

import logging
import os

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from .state import Evaluation, GraphState

logger = logging.getLogger(__name__)


def _make_llm() -> ChatAnthropic:
    """Construye el chat model de Claude.

    Por qué una función y no una constante global: para leer GD_MODEL DESPUÉS de que
    main.py haya corrido load_dotenv(). Si lo instanciáramos al importar el módulo,
    el .env todavía no estaría cargado.
    """
    model = os.getenv("GD_MODEL", "claude-sonnet-4-6")
    # temperature alta-ish en el generator favorece variedad entre iteraciones;
    # acá usamos un valor medio que sirve para ambos nodos.
    return ChatAnthropic(model=model, temperature=0.7, max_tokens=1024)


# ---------------------------------------------------------------------------
# NODO 1: GENERATOR
# ---------------------------------------------------------------------------
def generator_node(state: GraphState) -> dict:
    """Genera (o MEJORA) el draft.

    - Iteración 1 (sin feedback todavía): genera desde cero a partir de la task.
    - Iteraciones siguientes: recibe el feedback del discriminator y MEJORA el draft.

    Este "recibo feedback y vuelvo a generar" es exactamente el lado izquierdo de
    tu Ralph Loop en Barto-MCP. La diferencia: acá no hay un `while`, lo orquesta
    el grafo vía el edge condicional (ver graph.py).
    """
    llm = _make_llm()
    next_iteration = state.iteration + 1

    system = SystemMessage(
        content=(
            "Sos un escritor experto. Producí contenido claro, preciso y bien estructurado. "
            "Devolvé SOLO el contenido pedido, sin preámbulos ni meta-comentarios."
        )
    )

    if state.feedback:
        # Hay feedback de una ronda anterior -> modo MEJORA.
        human = HumanMessage(
            content=(
                f"Tarea: {state.task}\n\n"
                f"Tu borrador anterior fue:\n{state.draft}\n\n"
                f"Un evaluador lo calificó {state.score}/100 con este feedback:\n{state.feedback}\n\n"
                "Reescribí el contenido aplicando el feedback para subir la calidad."
            )
        )
        mode = "improve"
    else:
        # Primera pasada -> modo GENERACIÓN desde cero.
        human = HumanMessage(content=f"Tarea: {state.task}\n\nEscribí el contenido pedido.")
        mode = "generate"

    response = llm.invoke([system, human])
    draft = response.content if isinstance(response.content, str) else str(response.content)

    # Logging de la transición (requisito del spec): qué nodo corrió + en qué iteración.
    logger.info("[generator] mode=%s iteration=%d -> draft de %d chars", mode, next_iteration, len(draft))

    # Devolvemos el PARCHE. No tocamos score/feedback acá: eso es laburo del discriminator.
    return {
        "draft": draft,
        "iteration": next_iteration,
    }


# ---------------------------------------------------------------------------
# NODO 2: DISCRIMINATOR
# ---------------------------------------------------------------------------
def discriminator_node(state: GraphState) -> dict:
    """Evalúa el draft y devuelve score + feedback ESTRUCTURADOS.

    Acá está el requisito duro: with_structured_output(Evaluation). El LLM queda
    obligado a devolver un objeto Evaluation válido (score 0-100 + feedback), no un
    string suelto. En Barto-MCP esto lo parseabas a mano; acá lo garantiza el framework.
    """
    llm = _make_llm()
    # .with_structured_output() envuelve al modelo: en vez de texto, devuelve una
    # instancia de Evaluation (usa tool-calling de Anthropic por debajo).
    evaluator = llm.with_structured_output(Evaluation)

    system = SystemMessage(
        content=(
            "Sos un evaluador de calidad exigente pero justo. Calificás contenido de 0 a 100 "
            "según claridad, precisión, estructura y qué tan bien cumple la tarea pedida. "
            "Sé concreto en el feedback: decí QUÉ mejorar, no generalidades."
        )
    )
    human = HumanMessage(
        content=(
            f"Tarea original: {state.task}\n\n"
            f"Contenido a evaluar:\n{state.draft}\n\n"
            "Calificá de 0 a 100 y dame feedback accionable."
        )
    )

    evaluation: Evaluation = evaluator.invoke([system, human])

    logger.info(
        "[discriminator] iteration=%d -> score=%d (threshold=%d)",
        state.iteration,
        evaluation.score,
        state.threshold,
    )

    # Acumulamos el snapshot de esta ronda en history (leer + append + devolver lista nueva).
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
