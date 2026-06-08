"""Construcción del StateGraph: nodos + edges + ciclo + checkpointing.

CONCEPTOS CLAVE (#3 y #4 del modelo mental):
- Edges = el flujo de control. Hay fijos (add_edge) y condicionales (add_conditional_edges).
- El edge condicional es lo que crea el CICLO: vuelve de discriminator a generator.
  Eso es lo que una chain lineal de LangChain NO puede hacer. Es la razón de ser de LangGraph.
"""

from __future__ import annotations

import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .nodes import discriminator_node, generator_node
from .state import GraphState

logger = logging.getLogger(__name__)


def should_continue(state: GraphState) -> str:
    """La FUNCIÓN DE DECISIÓN del edge condicional. El cerebro del loop.

    Mira el State y devuelve una etiqueta ("end" o "generator"). LangGraph usa esa
    etiqueta para saber a qué nodo saltar. Esto reemplaza, de forma DECLARATIVA, el
    `if (score >= threshold || iteration >= max) break;` que en Barto-MCP vivía
    enterrado dentro de un `while`.

    Dos condiciones de corte (la segunda es el guard anti-loop-infinito del spec):
      1. score >= threshold  -> el draft ya está lo bastante bueno.
      2. iteration >= max_iterations -> agotamos los intentos; cortamos igual.
    """
    if state.score >= state.threshold:
        logger.info("[router] score %d >= threshold %d -> END", state.score, state.threshold)
        return "end"

    if state.iteration >= state.max_iterations:
        logger.info(
            "[router] iteration %d >= max %d -> END (guard anti loop infinito)",
            state.iteration,
            state.max_iterations,
        )
        return "end"

    logger.info("[router] score %d < threshold -> vuelvo a GENERATOR con el feedback", state.score)
    return "generator"


def build_graph():
    """Arma y compila el grafo. Devuelve una app ejecutable (.invoke / .stream)."""

    # 1) Creamos el grafo declarando QUÉ schema de estado usa. Así LangGraph sabe
    #    cómo hidratar/validar el State en cada nodo (lo convierte a GraphState).
    workflow = StateGraph(GraphState)

    # 2) Registramos los nodos. Nombre lógico -> función. (add_node del spec)
    workflow.add_node("generator", generator_node)
    workflow.add_node("discriminator", discriminator_node)

    # 3) Punto de entrada: por dónde arranca el grafo. (set_entry_point del spec)
    workflow.set_entry_point("generator")

    # 4) Edge FIJO: después del generator SIEMPRE vamos al discriminator.
    #    No hay decisión que tomar acá: generaste -> te evalúan.
    workflow.add_edge("generator", "discriminator")

    # 5) Edge CONDICIONAL: el ciclo. (add_conditional_edges del spec)
    #    Después del discriminator corremos should_continue(state). Según devuelva:
    #      "generator" -> volvemos a generar (CIERRA EL CICLO)
    #      "end"       -> terminamos
    #    El tercer argumento es el "mapa": etiqueta -> destino real.
    workflow.add_conditional_edges(
        "discriminator",
        should_continue,
        {
            "generator": "generator",  # <- esta flecha de vuelta es el CICLO
            "end": END,                # END es el nodo terminal especial de LangGraph
        },
    )

    # 6) Checkpointing (requisito del spec): MemorySaver guarda un snapshot del State
    #    después de CADA nodo, en memoria, indexado por thread_id. Esto te da:
    #      - persistencia del estado entre pasos
    #      - poder reanudar / inspeccionar el historial de la corrida
    #      - base para human-in-the-loop y time-travel
    #    (En prod usarías SqliteSaver/PostgresSaver; MemorySaver es para demo/tests.)
    checkpointer = MemorySaver()

    # 7) compile() congela la topología y nos da la app ejecutable.
    return workflow.compile(checkpointer=checkpointer)
