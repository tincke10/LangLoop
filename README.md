# langgraph-generator-discriminator

Reimplementación en **LangGraph** del patrón **Generator–Discriminator (Ralph Loop)**:
un loop iterativo de mejora de calidad donde un generador produce contenido y un
discriminador lo evalúa, repitiendo hasta alcanzar un umbral de calidad o un tope de
iteraciones.

Es la versión "con framework" de un patrón que originalmente está escrito **a mano en
TypeScript** en el repo `Barto-MCP`. La gracia de este repo es mostrar **qué te da
LangGraph gratis** que en la versión manual tenías que orquestar vos.

---

## La idea en una imagen

```
                    ┌──────────────────┐
        ┌──────────►│    GENERATOR     │
        │           │  (genera/mejora) │
        │           └────────┬─────────┘
        │                    │  edge fijo: generaste -> te evalúan
        │                    ▼
        │           ┌──────────────────┐
        │           │  DISCRIMINATOR   │
        │           │ (score+feedback) │
        │           └────────┬─────────┘
        │                    │
        │           should_continue(state)   ← edge CONDICIONAL (el router)
        │                    │
        │     ┌──────────────┴───────────────┐
        │     │                              │
   score < threshold                  score >= threshold
   y quedan iters                          O iter >= max
        │     │                              │
        └─────┘                              ▼
   (CICLO: vuelve con                      [END]
    el feedback)
```

La flecha que **vuelve** de `discriminator` a `generator` es el **ciclo**. Eso es lo
que una chain lineal de LangChain **no puede** modelar — y la razón por la que existe
LangGraph.

---

## Arquitectura del repo

```
.
├── pyproject.toml                      # deps + config (uv)
├── env.example                         # plantilla de variables (copiá a .env)
├── src/generator_discriminator/
│   ├── state.py                        # GraphState (Pydantic) + Evaluation (structured output)
│   ├── nodes.py                        # generator_node + discriminator_node
│   ├── graph.py                        # StateGraph: nodos, edges, ciclo, MemorySaver
│   └── main.py                         # entrypoint del demo
└── README.md
```

Responsabilidad de cada archivo:

| Archivo     | Qué resuelve                                                                 |
|-------------|------------------------------------------------------------------------------|
| `state.py`  | El **State** compartido (store) + el schema tipado del structured output.    |
| `nodes.py`  | El **trabajo**: cada nodo es una función `(state) -> dict` (parche de estado).|
| `graph.py`  | El **flujo**: registra nodos, conecta edges, define el **ciclo** y el checkpointing. |
| `main.py`   | Carga `.env`, compila el grafo y lo corre con `invoke`.                      |

---

## Cómo correrlo

```bash
# 1. Instalar deps (uv resuelve Python 3.12 solo)
uv sync --python 3.12

# 2. Configurar la API key
cp env.example .env        # y completá ANTHROPIC_API_KEY

# 3. Correr el loop
uv run gd-loop
# o:  uv run python -m generator_discriminator.main
```

Vas a ver, en los logs, **cada transición**: qué nodo corrió, en qué iteración, con qué
score, y la decisión del router (seguir o cortar).

---

## Conceptos clave de LangGraph (repaso para la entrevista)

### 1. StateGraph
La estructura central. Un grafo dirigido donde declarás **nodos** (trabajo) y **edges**
(flujo). Le pasás un **schema de estado** (acá un modelo Pydantic) y él sabe cómo
hidratar/validar ese estado en cada paso. Se **compila** (`.compile()`) a una app
ejecutable con `.invoke()` / `.stream()`.

### 2. State (estado compartido)
Los nodos **no se pasan datos como argumentos**. Existe **un** objeto de estado que
fluye por todo el grafo. Cada nodo lo **lee** y devuelve un **parche** (dict) con los
campos a actualizar; LangGraph lo mergea. Mentalidad **Redux**: State = store, nodo =
reducer. Por defecto el merge **sobrescribe**; para **acumular** (ej. una lista) usás un
*reducer* (`Annotated[list, add]`) o lo hacés a mano leyendo+appendeando (lo que hacemos
con `history`).

### 3. Nodos
Una función `(state) -> dict`. Nada más. La **unidad de trabajo**. Devuelve solo los
campos que cambia, no el State entero.

### 4. Edges
El **flujo de control**, declarativo y separado de la lógica:
- **Fijo** (`add_edge`): "después de A, siempre B".
- **Condicional** (`add_conditional_edges`): corre una función `(state) -> etiqueta` y
  salta según el resultado. **Es lo que crea el ciclo.**

### 5. Ciclo vs chain lineal
Una chain de LangChain es un **DAG (acíclico)**: `A → B → C`, sin vuelta atrás. LangGraph
permite **ciclos**: un edge condicional puede mandarte de vuelta a un nodo anterior. Por
eso modela loops de refinamiento, agentes ReAct, human-in-the-loop, etc.

### 6. Guard de iteraciones
La condición de corte por `max_iterations` vive en la función del edge condicional
(`should_continue`). Sin ella, un draft que nunca supera el umbral generaría un **loop
infinito**. El guard es **obligatorio** en cualquier grafo cíclico.

### 7. Checkpointing
`MemorySaver` guarda un **snapshot del State después de cada nodo**, indexado por
`thread_id`. Habilita persistencia, reanudar una corrida, inspeccionar el historial,
time-travel y human-in-the-loop. En prod se usa `SqliteSaver` / `PostgresSaver`.

### 8. Structured output
`llm.with_structured_output(Evaluation)` obliga al modelo a devolver un objeto Pydantic
válido (acá `{score, feedback}`) usando tool-calling del provider. Adiós a parsear
strings a mano.

---

## Barto-MCP (a mano) vs LangGraph (este repo)

| En Barto-MCP lo hacías así (a mano)                        | En LangGraph se modela así                                  |
|------------------------------------------------------------|-------------------------------------------------------------|
| Objeto que pasabas y mutabas entre funciones               | **State** Pydantic, mergeado por el framework               |
| `generator()` y `discriminator()` como funciones sueltas   | **Nodos** registrados con `add_node`                        |
| `while (...) { ... if (cond) break; }`                     | **Edge condicional** `add_conditional_edges` + `should_continue` |
| El `if (score >= th || iter >= max) break;` dentro del while| La **función del router**, declarativa y separada           |
| Parsear el texto del discriminator a mano                  | `with_structured_output(Evaluation)` tipado y garantizado   |
| (no tenías) persistencia/reanudación del estado            | **Checkpointing** con `MemorySaver`                         |
| `console.log` desperdigados                                | Logging por transición en cada nodo + router                |

El punto para la entrevista: **el patrón es el mismo; LangGraph lo formaliza y te da
gratis topología explícita, checkpointing y structured output.**

---

## 5 preguntas de entrevista (con respuestas)

**1. ¿Por qué LangGraph y no una chain de LangChain para este problema?**
Porque una chain es un grafo **acíclico** (DAG): va de A a B a C sin volver atrás. Este
problema necesita un **ciclo** — generar, evaluar, y si no alcanza el umbral, **volver**
a generar con el feedback. LangGraph soporta edges condicionales que cierran ese ciclo;
una chain no. Lo ves en `graph.py`: el edge condicional mapea `"generator" ->
"generator"`, esa flecha de vuelta es el loop.

**2. ¿Cómo se comunican los nodos entre sí si no se pasan argumentos?**
A través del **State compartido**. Cada nodo recibe el State completo, hace su trabajo y
devuelve un **dict parche** con los campos que actualiza; LangGraph lo mergea al State.
Es el modelo Redux: el State es el store, cada nodo un reducer. El `generator` escribe
`draft`; el `discriminator` lee ese `draft` del State y escribe `score`/`feedback`.

**3. ¿Cómo evitás un loop infinito?**
Con un **guard** en la función del edge condicional (`should_continue`). Tiene dos cortes:
`score >= threshold` (logramos calidad) **o** `iteration >= max_iterations` (agotamos
intentos). El `generator` incrementa `iteration` en cada pasada, así que el segundo corte
siempre se cumple eventualmente. Sin ese guard, un draft que nunca supere el umbral
correría para siempre.

**4. ¿Para qué sirve el checkpointer (`MemorySaver`) si el grafo ya corre solo?**
Persiste un **snapshot del State después de cada nodo**, indexado por `thread_id`. Eso
habilita: reanudar una corrida interrumpida, inspeccionar el historial paso a paso,
hacer *time-travel* (volver a un estado anterior) y *human-in-the-loop* (pausar, dejar
que un humano edite el State, y seguir). `MemorySaver` es en memoria para demo/tests; en
producción usás `SqliteSaver` o `PostgresSaver`. Por eso `invoke` requiere un
`thread_id` en `config` cuando hay checkpointer.

**5. ¿Cómo garantizás que el discriminator devuelva un score numérico y no texto libre?**
Con `llm.with_structured_output(Evaluation)`, donde `Evaluation` es un modelo Pydantic
con `score: int (0-100)` y `feedback: str`. Por debajo usa el **tool-calling** del
provider para forzar al modelo a producir un objeto que matchee el schema; Pydantic
valida tipos y rangos. Si el modelo intentara devolver algo inválido, falla la
validación en vez de propagarse un dato corrupto. Es la diferencia con parsear el string
a mano, que es frágil.

---

## Stack

- **Python 3.12** + **uv**
- **langgraph** (StateGraph, edges condicionales, MemorySaver)
- **langchain-anthropic** (`ChatAnthropic`, `with_structured_output`)
- **pydantic** (State tipado + structured output)
- **python-dotenv** (carga de `ANTHROPIC_API_KEY`)
