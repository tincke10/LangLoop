"""State tipado del grafo + schema de salida estructurada del discriminator.

CONCEPTO CLAVE (#1 del modelo mental):
En LangGraph los nodos NO se pasan datos como argumentos de función. Existe UN
objeto de estado compartido que vive durante toda la corrida del grafo. Cada nodo
lo lee y devuelve un "parche" (dict) con los campos que quiere actualizar; LangGraph
mergea ese parche al estado. Es Redux: el State es el store, cada nodo es un reducer.

Por qué Pydantic y no un TypedDict:
- Validación en runtime (score 0-100 garantizado, no un int cualquiera).
- Defaults declarativos (iteration arranca en 0 sin que nadie lo setee).
- Es el MISMO modelo que reusamos para el structured output del discriminator.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Evaluation(BaseModel):
    """Salida ESTRUCTURADA del discriminator.

    Este modelo es el contrato que le pasamos a `llm.with_structured_output(Evaluation)`.
    El provider (Claude) queda OBLIGADO a devolver exactamente {score, feedback} con
    los tipos correctos. Adiós a parsear un string a mano rezando que venga el JSON bien.

    En Barto-MCP esto lo hacías a mano: el discriminator devolvía texto y vos lo
    parseabas. Acá el framework + Pydantic te lo garantizan tipado.
    """

    score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Calidad del draft de 0 a 100. 100 = perfecto, cumple todos los criterios.",
    )
    feedback: str = Field(
        ...,
        description=(
            "Crítica concreta y accionable: qué falta o qué mejorar para subir el score. "
            "Si el score ya es alto, explicá brevemente por qué está bien."
        ),
    )


class GraphState(BaseModel):
    """El estado compartido que fluye por TODO el grafo.

    Cada campo es un "canal" que los nodos leen y escriben. Mirá cómo cada uno
    mapea 1:1 con un requisito del spec.
    """

    # --- Input del usuario ---
    task: str = Field(..., description="La consigna. Ej: 'escribí un párrafo sobre X'.")

    # --- Lo que produce el generator ---
    draft: str = Field(default="", description="El contenido generado en la última iteración.")

    # --- Lo que produce el discriminator (structured output) ---
    score: int = Field(default=0, description="Último score 0-100 dado por el discriminator.")
    feedback: str = Field(default="", description="Último feedback. El generator lo usa para mejorar.")

    # --- Control del loop ---
    iteration: int = Field(default=0, description="Cuántas veces corrió el generator. El guard lo usa.")
    max_iterations: int = Field(default=5, description="Tope duro de iteraciones. Evita loops infinitos.")
    threshold: int = Field(default=85, description="Score mínimo para cortar y dar el draft por bueno.")

    # --- Trazabilidad ---
    # Acumulamos un snapshot por iteración para poder mostrar la evolución del loop.
    # Lo manejamos a mano en los nodos (leer history, append, devolver la lista nueva)
    # en vez de usar un reducer Annotated: más explícito y más fácil de defender.
    history: list[dict] = Field(
        default_factory=list,
        description="Un registro por iteración: {iteration, score, feedback}.",
    )
