"""Entrypoint del demo. Carga el .env, arma el grafo y corre el loop.

Corré con:  uv run gd-loop
        o:  uv run python -m generator_discriminator.main
"""

from __future__ import annotations

import logging
import os
import sys

from dotenv import load_dotenv

from .graph import build_graph


def _configure_logging() -> None:
    """Logging básico de cada transición (requisito del spec)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    # 1) Cargar ANTHROPIC_API_KEY (y GD_MODEL si está) desde .env ANTES de tocar el LLM.
    load_dotenv()
    _configure_logging()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: falta ANTHROPIC_API_KEY. Copiá env.example a .env y completá la key.")
        sys.exit(1)

    # 2) Armar la app (grafo compilado con checkpointer).
    app = build_graph()

    # 3) thread_id: identifica esta corrida en el checkpointer. Con el mismo thread_id
    #    podrías reanudar/inspeccionar el estado guardado. Es OBLIGATORIO cuando hay
    #    checkpointer compilado.
    config = {"configurable": {"thread_id": "demo-1"}}

    # 4) Estado inicial. Solo seteamos lo que el usuario aporta; el resto usa los
    #    defaults del modelo Pydantic (draft="", score=0, iteration=0, ...).
    initial_state = {
        "task": "Escribí un párrafo claro y atractivo explicando qué es LangGraph "
        "y por qué sirve para flujos cíclicos con LLMs.",
        "max_iterations": 5,
        "threshold": 85,
    }

    print("=" * 70)
    print("GENERATOR–DISCRIMINATOR LOOP (Ralph Loop en LangGraph)")
    print("=" * 70)
    print(f"Tarea: {initial_state['task']}\n")

    # 5) invoke() corre el grafo hasta llegar a END y devuelve el State final (dict).
    final_state = app.invoke(initial_state, config=config)

    # 6) Reporte de la corrida.
    print("\n" + "=" * 70)
    print("EVOLUCIÓN DEL LOOP")
    print("=" * 70)
    for entry in final_state["history"]:
        print(f"  iteración {entry['iteration']}: score={entry['score']}")
        print(f"      feedback: {entry['feedback']}")

    print("\n" + "=" * 70)
    print(f"RESULTADO FINAL  (score={final_state['score']}, iteraciones={final_state['iteration']})")
    print("=" * 70)
    print(final_state["draft"])


if __name__ == "__main__":
    main()
