# Academic Performance Analysis

Laboratorio de producción para construir un producto analítico reproducible
a partir de datos académicos seudonimizados. Sirve como material práctico
de la **Unidad 4** ("Despliegue de modelos de Inteligencia Artificial") del
curso *Técnicas avanzadas de IA*.

El repositorio avanza en **un solo lugar**, mediante commits sucesivos: no
hay ramas paralelas ni entregas separadas. Cada etapa queda marcada con un
**tag** de Git para poder ver exactamente el estado del proyecto en ese
punto, sin tener que leer todo el historial.

## Cómo navegar el historial

```bash
# Ver todas las etapas disponibles, en orden
git tag -l -n1

# Pararse en el estado del proyecto en una etapa específica
git checkout <tag>

# Volver a la última versión
git checkout main
```

| Tag | Qué muestra |
|---|---|
| `lab0-esqueleto` | Estructura inicial del proyecto: `pyproject.toml`, esquema de datos, carpetas vacías. |
| `lab1-paso1-ingesta-validacion-preparacion` | Pipeline de datos: ingesta con SHA-256, validación contra `config/data_schema.yaml`, preparación. |
| `lab1-paso2-analisis-estadistico` | Prueba de hipótesis (t de Welch): ¿la asistencia se asocia con la nota? |
| `lab2-paso1-modelo-regresion` | Primer modelo real (regresión), con seguimiento de experimentos en MLflow. |
| `lab2-paso2-modelo-clasificacion` | Segundo modelo (clasificación: ¿aprueba?), manifiesto extendido para varios modelos. |
| `lab3-paso1-comparacion-seleccion` | Un segundo algoritmo candidato por tarea + selección con un baseline ingenuo explícito. |
| `lab3-paso2-model-registry` | Los modelos seleccionados (y el rechazado) quedan en el MLflow Model Registry. |
| `lab4-servicio-inferencia` | El modelo candidato se empaqueta, se sirve con FastAPI y se conteneriza con Docker. |
| `pruebas-automatizadas` | Pruebas reales con pytest sobre el pipeline y el API (antes eran placeholders). Corrige un bug real que estas pruebas encontraron en `api.py`. |

Cada tag es un punto donde **todo corre**: `uv sync`, `pytest`, y los
comandos de esa sección del README funcionan tal como están documentados
en ese momento del historial.

---

## Laboratorio 1 — Ingesta, validación y preparación

*(tags `lab1-paso1-...` y `lab1-paso2-...`)*

Flujo inicial:

`data/incoming -> ingest -> metadata -> validate -> data/validated -> prepare -> data/prepared -> analysis -> results`

Cada etapa deja un rastro verificable (SHA-256, conteos, estado) en
`metadata/<dataset_id>.json`, que las etapas siguientes comprueban antes de
continuar — así una etapa nunca opera sobre datos que no pasaron por la
anterior.

```bash
# Ingesta
uv run python -m academic_analysis.ingest --file data/incoming/academic_performance_ING-20260910-101728.csv

# Validación
uv run python -m academic_analysis.validate --dataset ING-20260910-101728

# Preparación
uv run python -m academic_analysis.prepare --dataset ING-20260910-101728

# Análisis (prueba de hipótesis: asistencia vs. nota)
uv run python -m academic_analysis.analysis --dataset ING-20260910-101728
```

El análisis responde una pregunta puntual: ¿existe una diferencia
estadística entre la nota de quienes asisten y quienes no? **No es un
modelo predictivo** — es la motivación para el Laboratorio 2.

## Laboratorio 2 — Primeros modelos predictivos

*(tags `lab2-paso1-...` y `lab2-paso2-...`)*

Los modelos de este laboratorio usan asistencia, curso y semestre para
predecir:

- **Regresión**: la nota exacta del curso.
- **Clasificación**: si el estudiante aprueba (`Nota Curso >= 3.0`).

Para cada tarea hay dos algoritmos candidatos (ver `--model-type` abajo).
Todos son una primera aproximación, con desempeño modesto —ver
`notebooks/Unidad4_lab2.ipynb` para la interpretación completa, incluyendo
por qué un *accuracy* del 78% en clasificación en realidad no es tan bueno
como parece.

Cada ejecución queda registrada en MLflow (parámetros, métricas y el modelo
como artefacto) y también resumida en `metadata/<dataset_id>.json`, bajo
`training.regression.<model_type>` y `training.classification.<model_type>`,
junto al `run_id` correspondiente de cada una.

```bash
# Entrenamiento (regresión: linear o random_forest)
uv run python -m academic_analysis.train regression --model-type linear --dataset ING-20260910-101728
uv run python -m academic_analysis.train regression --model-type random_forest --dataset ING-20260910-101728

# Entrenamiento (clasificación: logistic o random_forest)
uv run python -m academic_analysis.train classification --model-type logistic --dataset ING-20260910-101728
uv run python -m academic_analysis.train classification --model-type random_forest --dataset ING-20260910-101728

# Explorar los experimentos registrados
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

`mlruns/` y `mlflow.db` son estado local de MLflow: se reconstruyen
ejecutando `train.py` y no se versionan en Git. Lo que sí se versiona es
cada modelo entrenado (`models/`), la matriz de confusión (`results/`) y el
resumen de cada experimento en el manifiesto.

## Laboratorio 3 — Comparación, selección y registro

*(tags `lab3-paso1-...` y `lab3-paso2-...`)*

### Comparar y seleccionar

Comparar los candidatos entrenados y seleccionar uno, con un criterio
explícito: superar un **baseline ingenuo** (predecir siempre el promedio, o
predecir siempre la clase mayoritaria). Ver
`notebooks/Unidad4_lab3_seleccion.ipynb` para la lectura completa.

```bash
uv run python -m academic_analysis.compare regression --dataset ING-20260910-101728
uv run python -m academic_analysis.compare classification --dataset ING-20260910-101728
```

El resultado (criterio, baseline, cada candidato y cuál —si alguno— fue
seleccionado) queda en `metadata/<dataset_id>.json`, bajo
`selection.regression` / `selection.classification`. En este dataset, la
regresión sí encuentra un candidato válido; en clasificación, **ningún
candidato supera el baseline todavía**, así que no se selecciona ninguno —
es el comportamiento esperado, no un error.

### Model Registry

Registrar en MLflow lo que decidió la comparación anterior, con un alias
por estado:

- `candidato`: superó el baseline (regresión).
- `archivado`: no lo superó, pero queda registrado con la razón exacta del
  rechazo (clasificación) — el registry conserva también el historial de
  lo descartado, no solo lo que funcionó.

```bash
uv run python -m academic_analysis.registry regression --dataset ING-20260910-101728
uv run python -m academic_analysis.registry classification --dataset ING-20260910-101728
```

Requiere haber corrido antes `academic_analysis.compare` para esa tarea.
El resultado queda en `metadata/<dataset_id>.json`, bajo
`registry.regression` / `registry.classification`, y es visible en la
pestaña *Models* de `mlflow ui`.

## Laboratorio 4 — Empaquetar y servir el modelo

*(tag `lab4-servicio-inferencia`)*

Todo lo anterior (ingesta, validación, entrenamiento, MLflow, el
manifiesto completo) es información de **desarrollo**: necesaria para
decidir qué modelo usar, pero no para el servicio que finalmente responde
solicitudes. Este laboratorio separa explícitamente ambos mundos:

```text
DESARROLLO                         DESPLIEGUE
(todo el repo)                     (solo deploy/ + api.py)

datos, manifiesto completo    →    (no viaja)
MLflow (mlflow.db, mlruns/)   →    (no viaja)
modelo rechazado              →    (no viaja)
notebooks                     →    (no viaja)
modelo candidato + métricas   →    deploy/model.joblib
                                    deploy/model_info.json
```

### 1. Empaquetar

Extrae del registro el modelo con alias `candidato` y genera el paquete
mínimo de despliegue:

```bash
uv run python -m academic_analysis.package --dataset ING-20260910-101728
```

Genera `deploy/model.joblib` (pesos + columnas + categorías válidas) y
`deploy/model_info.json` (versión, alias, métricas — nada sobre el
dataset fuente, el `run_id`, ni el modelo rechazado).

### 2. Servir con FastAPI

`academic_analysis.api` es un módulo independiente: no importa nada de
`train.py` / `compare.py` / `registry.py`, así que no necesita MLflow en
tiempo de ejecución — solo lee `deploy/`.

```bash
uv run uvicorn academic_analysis.api:app --reload
```

Endpoints:

- `GET /health` — confirma que el servicio y el modelo cargaron.
- `GET /model-info` — versión, alias y métricas del modelo servido.
- `POST /predict` — recibe `{"n_asistencias": 5, "Curso": "...", "Semestre": "2026-1"}`
  y devuelve la nota predicha. Valida `Curso`/`Semestre` contra las
  categorías vistas en entrenamiento (Unidad 4, Cap. 4: "Validar la
  entrada") y responde `422` con un mensaje claro si no reconoce alguna.

### 3. Contenedor

```bash
docker build -t academic-performance-api .
docker run --rm -p 8000:8000 academic-performance-api
```

El `Dockerfile` usa `uv sync --no-default-groups`, que instala **solo**
`fastapi`, `uvicorn`, `pandas` y `scikit-learn` (lo declarado en
`[project.dependencies]`) — nada de `mlflow`, `matplotlib`, `scipy`,
`pyyaml`, ni las herramientas de desarrollo (`jupyter`, `pytest`, `ruff`).
Esas quedan agrupadas en `pyproject.toml` bajo los grupos `pipeline` y
`dev`, que `uv sync` local sigue instalando por defecto (ver
`[tool.uv] default-groups`), pero que la imagen nunca ve. La imagen
tampoco copia `data/`, `metadata/`, `notebooks/`, `mlruns/` ni
`mlflow.db` (ver `.dockerignore`) — solo `deploy/` y el módulo `api.py`.

### 4. Mostrarlo públicamente (túnel temporal)

Para que alguien fuera de la red local pueda probar el servicio sin
desplegar infraestructura permanente, usar un túnel efímero de
[Cloudflare](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/):

```bash
# instalar cloudflared una sola vez, luego:
cloudflared tunnel --url http://localhost:8000
```

Imprime una URL pública temporal (`https://algo-al-azar.trycloudflare.com`)
que reenvía tráfico al contenedor mientras el comando siga corriendo. No
requiere cuenta ni configuración — apto para una demostración en clase,
no para dejarlo corriendo de forma permanente.

---

## Pruebas automatizadas

*(tag `pruebas-automatizadas`)*

```bash
uv run pytest -v
```

Cubren la lógica pura del pipeline (sin necesitar datos reales en disco ni
MLflow corriendo): parseo del `dataset_id`, validaciones de esquema
(`validate.py`), normalización y deduplicación de registros (`prepare.py`),
construcción de features/target para ambos modelos (`train.py`), y la
codificación de entrada del servicio (`api.py`).

Ese último grupo de pruebas encontró un **bug real** mientras se escribía:
`_build_model_input` usaba `pd.get_dummies` sobre una sola fila, y con un
único valor presente en la fila, `drop_first` lo descarta *siempre* —
así que casi cualquier curso quedaba codificado como si fuera la categoría
de referencia de entrenamiento, sin importar cuál fuera. Una prueba manual
con `curl` no lo había detectado porque, por casualidad, probó justo con
el curso que sí era la referencia real. Corregido: las columnas se
construyen explícitamente a partir de `feature_columns`, sin depender de
cuántas categorías distintas trae la solicitud.
