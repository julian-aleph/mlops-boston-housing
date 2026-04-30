# MLOps local - Boston Housing

Proyecto MLOps local, open-source y agnóstico de nube para regresión sobre
Boston Housing, con pipeline reproducible, tracking de experimentos,
versionamiento de artefactos, API REST, despliegue local, monitoreo básico y
CI/CD.

## 1. Objetivo

- No es un ejercicio de entrenamiento aislado: implementa un flujo ML mínimo
  con orientación a producción.
- Cubre preparación de datos, entrenamiento, evaluación, persistencia,
  serving, monitoreo y reentrenamiento.
- Diseñado para correr completamente en una sola máquina, sin depender de
  servicios administrados.
- Las decisiones técnicas son agnósticas de proveedor y portables a nube sin
  reescribir el código.

## 2. Arquitectura general

```text
Raw data
  -> Data profile
  -> Feature preparation
  -> Training + MLflow
  -> Evaluation
  -> Promotion
  -> FastAPI serving
  -> Docker deployment
  -> Prometheus/Grafana monitoring
  -> GitHub Actions validation
```

Responsabilidades por componente:

- **Git:** código fuente, configuración y documentación.
- **Makefile:** interfaz operativa única para todas las tareas comunes.
- **DVC:** DAG del pipeline y versionamiento de artefactos generados.
- **MLflow:** tracking de experimentos con backend SQLite local.
- **joblib:** persistencia local del modelo final servido por la API.
- **FastAPI:** inferencia online con `/predict`, `/health` y `/metrics`.
- **Docker Compose:** despliegue local del stack `api + prometheus + grafana`.
- **Prometheus / Grafana:** observabilidad básica del servicio.
- **GitHub Actions:** validación automática en cada push y pull request.

## 3. Estructura del repositorio

```text
boston-housing-mlops/
├── app/                  # FastAPI: main, schemas, model loader, monitoring
├── src/                  # Pipeline: data, features, preprocessing, train, evaluate, promote
├── tests/                # Pytest: contratos de API, training y preprocessing
├── configs/              # params.yaml: configuración única de pipeline y serving
├── data/                 # raw/ y processed/ (no comiteados)
├── models/               # staging/ y production/ (no comiteados)
├── reports/              # métricas, perfiles, importancia de features
├── monitoring/           # prometheus.yml + provisioning de Grafana
├── docker/               # Dockerfile de la imagen serving
├── .github/workflows/    # ci.yml: workflow de CI
├── docs/                 # outline de presentación, checklist de validación
├── Makefile
├── docker-compose.yml
├── dvc.yaml
├── requirements.txt
├── pyproject.toml
└── README.md
```

Carpetas clave:

- `app/`: solo lo necesario para servir.
- `src/`: lógica del pipeline reutilizable y testeable.
- `configs/params.yaml`: punto único de configuración.
- `models/production/`: artefacto cargado por la API en runtime.

## 4. Requisitos

- Python 3.11
- Docker y Docker Compose
- Make
- Git
- Opcional: CLI de DVC si se ejecutan comandos `dvc` directamente

## 5. Instalación local

```bash
python3.11 -m venv .venv
source .venv/bin/activate
make setup
```

`make setup` instala las dependencias fijadas en `requirements.txt`.

## 6. Ruta rápida

Para validar el proyecto en pocos pasos sin entrar al detalle de cada
componente:

```bash
make setup
make pipeline
make serve
```

Esta secuencia:

1. Instala dependencias.
2. Ejecuta el pipeline completo y deja un modelo en `models/production/`.
3. Levanta la API en `http://localhost:8000`.

Para validar también el despliegue en contenedor y el monitoreo:

```bash
make pipeline
make docker-build
make docker-up
make monitor
```

Esto deja disponibles la API, Prometheus y Grafana en local. La sección
"Validación rápida final" detalla los checks manuales adicionales.

## 7. Comandos principales

Tabla resumen de los targets más usados del Makefile:

| Comando             | Propósito                                                  |
| ------------------- | ---------------------------------------------------------- |
| `make setup`        | Instala dependencias del proyecto.                         |
| `make pipeline`     | Ejecuta el pipeline end-to-end y promueve el modelo.       |
| `make retrain`      | Reentrena reejecutando el pipeline completo.               |
| `make serve`        | Levanta la API FastAPI local.                              |
| `make api-check`    | Valida que la API carga los artefactos productivos.        |
| `make mlflow-ui`    | Abre la UI de MLflow.                                      |
| `make docker-build` | Construye la imagen de serving.                            |
| `make docker-up`    | Levanta el stack `api + prometheus + grafana` local.       |
| `make docker-down`  | Detiene el stack local.                                    |
| `make monitor`      | Imprime URLs y credenciales del stack de observabilidad.   |
| `make lint`         | Ejecuta linting y validación de imports.                   |
| `make test`         | Ejecuta la suite de pruebas.                               |

Las secciones siguientes describen cada bloque en detalle.

## 8. Ejecución del pipeline con Makefile

El Makefile es la interfaz operativa principal. Todos los comandos del
pipeline están expuestos como targets.

Comandos individuales:

```bash
make data
make features
make train
make evaluate
make promote
```

- `make data`: descarga el dataset, valida el raw y genera el data profile.
- `make features`: normaliza columnas y produce dataset procesado, schema y
  registry de features.
- `make train`: entrena los candidatos y registra los runs en MLflow.
- `make evaluate`: evalúa el modelo staging y calcula importancia por
  permutación.
- `make promote`: promueve staging a producción si se cumplen las reglas de
  promoción.

Pipeline completo end-to-end:

```bash
make pipeline
```

Equivalente a:

```text
data -> features -> train -> evaluate -> promote
```

Reentrenamiento:

```bash
make retrain
```

`make retrain` reejecuta el pipeline completo. Se utiliza cuando hay nuevos
datos o cambios en la configuración.

## 9. Preparación de datos y features

- Dataset descargado desde KaggleHub (`altavish/boston-housing-dataset`).
- Variable objetivo: `MEDV`.
- El profiling raw produce `reports/metrics/data_profile.json`.
- La preparación de features genera:
  - `data/processed/boston_housing_processed.parquet`
  - `data/processed/feature_schema.json`
  - `data/processed/feature_registry.json`

No se implementa un feature store completo porque el dataset es pequeño y
estático. Para online/offline feature consistency se evaluaría Feast en una
fase posterior.

## 10. Entrenamiento y selección del modelo

Modelos candidatos:

- Ridge
- ElasticNet
- RandomForestRegressor
- HistGradientBoostingRegressor

Selección:

- `RandomizedSearchCV` con CV configurable en `configs/params.yaml`.
- Métrica primaria: RMSE.
- Métricas secundarias: MAE y R2.
- El modelo ganador se persiste en `models/staging/` y se promueve a
  `models/production/` si supera el umbral de mejora.

Uso de `sklearn Pipeline`:

- Mantiene preprocesamiento y modelo en un único artefacto.
- Reduce diferencias entre training y serving.
- Hace que el `joblib` persistido sea autosuficiente para inferencia.

## 11. Tracking de experimentos con MLflow

```bash
make train
make mlflow-ui
```

```text
http://localhost:5000
```

- MLflow está integrado al stage de entrenamiento.
- Cada candidato se registra como un run independiente.
- Se trackean hiperparámetros, métricas y el tag del modelo seleccionado.
- Backend store: SQLite en `mlflow.db`.
- Artifact store: directorio local `mlruns/`.
- MLflow no reemplaza la persistencia con joblib.
- FastAPI no depende de MLflow para servir predicciones.

## 12. Versionamiento de artefactos con DVC

```bash
make dvc-repro
make dvc-status
make dvc-metrics
```

Comandos directos:

```bash
dvc repro
dvc status
dvc metrics show
dvc dag
```

- DVC formaliza el pipeline como un DAG en `dvc.yaml`.
- Los artefactos generados se versionan vía DVC y no se comitean directamente
  en Git.
- No se configura un remote en este challenge.
- Una mejora futura es agregar un remote local, SSH, S3, GCS o equivalente.

## 13. Serving local con FastAPI

```bash
make pipeline
make serve
make api-check
```

URLs:

```text
http://localhost:8000/docs
http://localhost:8000/health
```

Healthcheck:

```bash
curl http://localhost:8000/health
```

Predicción:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "crim": 0.00632,
    "zn": 18.0,
    "indus": 2.31,
    "chas": 0.0,
    "nox": 0.538,
    "rm": 6.575,
    "age": 65.2,
    "dis": 4.09,
    "rad": 1.0,
    "tax": 296.0,
    "ptratio": 15.3,
    "b": 396.9,
    "lstat": 4.98
  }'
```

Forma esperada de la respuesta:

```json
{
  "prediction": 24.8,
  "model_name": "random_forest",
  "model_stage": "production",
  "target": "medv",
  "features_used": ["crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", "rad", "tax", "ptratio", "b", "lstat"]
}
```

El valor exacto de `prediction` depende del modelo promovido y no es fijo.

`make api-check` valida que `app.main` importa correctamente cargando los
artefactos de `models/production/`.

## 14. Despliegue local con Docker

El despliegue Docker sirve un modelo ya promovido. La imagen no entrena: es
únicamente para inferencia. Por eso es **prerrequisito** ejecutar primero el
pipeline para producir el artefacto productivo.

Orden esperado:

```bash
make pipeline        # paso obligatorio: genera models/production/
make docker-build    # construye la imagen de serving
make docker-up       # levanta api + prometheus + grafana
make docker-logs     # logs del stack
make docker-down     # detiene el stack
```

Si `models/production/` está vacío, la API arrancará sin modelo cargado y
`/health` reportará un estado no listo. La solución es ejecutar
`make pipeline` antes de `make docker-up`.

Detalles del despliegue:

- La imagen Docker es solo para serving.
- El entrenamiento se ejecuta fuera de la imagen.
- El contenedor monta `models/production/` y `configs/` en read-only, de
  modo que un nuevo `make pipeline` en el host se refleja al reiniciar el
  contenedor.
- FastAPI no depende de MLflow ni DVC en runtime.

Healthcheck contra el contenedor:

```bash
curl http://localhost:8000/health
```

Predicción contra el contenedor:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "crim": 0.00632,
    "zn": 18.0,
    "indus": 2.31,
    "chas": 0.0,
    "nox": 0.538,
    "rm": 6.575,
    "age": 65.2,
    "dis": 4.09,
    "rad": 1.0,
    "tax": 296.0,
    "ptratio": 15.3,
    "b": 396.9,
    "lstat": 4.98
  }'
```

## 15. Monitoreo básico

```bash
make docker-up
make monitor
```

URLs:

```text
FastAPI docs:    http://localhost:8000/docs
Metrics:         http://localhost:8000/metrics
Prometheus:      http://localhost:9090
Grafana:         http://localhost:3000
```

Acceso por defecto a Grafana:

```text
user: admin
password: admin
```

Métricas Prometheus:

```bash
curl http://localhost:8000/metrics
```

Métricas expuestas:

- `prediction_requests_total`
- `prediction_errors_total`
- `prediction_latency_seconds`
- `prediction_value`
- `model_loaded`
- `model_info`

Esta capa cubre monitoreo operativo y distribución de predicciones.
Detección avanzada de drift queda como mejora futura.

## 16. Calidad de código

```bash
make lint
make test
```

- **Ruff:** linting y formateo.
- **compileall:** validación de imports y sintaxis sobre `src/` y `tests/`.
- **pytest:** contratos de preprocessing, training y API, incluyendo el
  endpoint `/metrics`.

## 17. CI/CD con GitHub Actions

Workflow: `.github/workflows/ci.yml`.

Triggers:

- `push` a `main`
- `pull_request` a `main`

Pasos:

```text
make setup
make lint
make pipeline
make test
make api-check
make docker-build
```

Alcance del CI:

- **No hay despliegue a nube.** El proyecto es local-first y agnóstico de
  proveedor; CI no publica imágenes ni aplica infraestructura.
- `docker-build` se ejecuta como **validación de readiness para deploy**:
  confirma que el `Dockerfile` es válido, que la imagen se construye con el
  modelo productivo dentro del workflow y que el artefacto serviría sin
  cambios en cualquier entorno con Docker.
- Docker Compose, Prometheus y Grafana no se ejecutan en CI; solo se
  construye la imagen.

Sobre KaggleHub:

- El pipeline en CI descarga el dataset público vía KaggleHub.
- Si aparecen rate limits, se pueden configurar `KAGGLE_USERNAME` y
  `KAGGLE_KEY` como secrets del repositorio.

## 18. Reentrenamiento

```bash
make retrain
```

- `make retrain` ejecuta el pipeline completo (`data -> features -> train ->
  evaluate -> promote`).
- El modelo entrenado queda en `models/staging/` antes de ser evaluado.
- La promoción a producción se controla por comparación de métricas en
  `src/promote.py` (umbral `promotion.min_rmse_improvement`).
- Un modelo recién entrenado no sustituye al de producción solo por ser nuevo:
  debe superar el umbral de mejora.

## 19. Seguridad, escalabilidad y limitaciones

Limitaciones intencionales para mantener el alcance del challenge:

- Sin autenticación ni autorización en los endpoints.
- Sin rate limiting.
- Sin detección avanzada de drift (datos, concept, performance).
- Sin gestión de secrets de grado producción.
- Sin MLflow Model Registry.
- Sin remote DVC configurado.
- Sin despliegue a nube.
- Sin estrategia canary ni blue/green.

Estas decisiones priorizan claridad y reproducibilidad en un entregable de
6 a 8 horas, sin acoplar el código a un proveedor específico.

## 20. Consideraciones éticas sobre el dataset

Boston Housing es un dataset histórico (1978) con limitaciones documentadas.
En particular, la variable `B` es una transformación de la proporción de
residentes afroamericanos por barrio, construida bajo supuestos del estudio
original que hoy no se consideran adecuados para sistemas de decisión.

Decisiones del proyecto:

- El dataset se utiliza únicamente como caso de demostración técnica del
  flujo MLOps, no como referencia para tasación inmobiliaria real.
- No se realiza análisis de fairness ni mitigación de sesgos, dado que el
  foco del entregable es la infraestructura y no el modelo.
- Para un caso productivo se recomendaría reemplazar el dataset por uno
  actual sin variables sensibles ni proxies de raza, y agregar revisiones
  de fairness antes de promover a producción.

## 21. Mejoras futuras

Componentes que no forman parte del proyecto actual y se evaluarían en
fases posteriores:

- **Feature store:** Feast si se requiere consistencia online/offline de
  features compartidas.
- **Infra como código (Terraform):** evolución posible si el proyecto se
  lleva a nube. No se incluye hoy porque el alcance es local-first; se
  introduciría junto con S3 / GCS para DVC, RDS / Cloud SQL para MLflow y
  ECS / EKS / Cloud Run para el API.
- **MLflow multi-usuario:** backend PostgreSQL en lugar de SQLite y artifact
  store remoto.
- **Model governance:** MLflow Model Registry con stages aprobados.
- **Drift monitoring:** Evidently o WhyLabs OSS para drift de datos y
  predicciones, con reentrenamiento disparado por alertas.
- **Hardening del API:** autenticación (OAuth2 / API keys) y rate limiting.
- **Despliegue progresivo:** canary o blue/green con un proxy.
- **Container registry:** push automatizado de la imagen a un registry
  privado.
- **Remote DVC:** almacenamiento compartido de artefactos versionados.

## 22. Uso de herramientas de IA

Se usaron herramientas de asistencia con IA como apoyo puntual para tareas
acotadas: planeación inicial, generación de boilerplate, redacción y
revisión de documentación, e ideación de casos de prueba. No se delegó la
toma de decisiones técnicas ni la validación del sistema. La definición
de la arquitectura, la integración entre componentes, la depuración, la
validación funcional y la revisión final fueron realizadas manualmente por
el autor. El proyecto es reproducible mediante los comandos documentados en
Makefile, DVC y Docker.

## 23. Validación rápida final

Comandos automatizables:

```bash
make setup
make pipeline
make api-check
make docker-build
make lint
make test
```

Servicios locales:

```bash
make docker-up
make monitor
```

Checks manuales:

- Abrir `http://localhost:8000/docs` y disparar `/predict`.
- Validar `/health` y `/metrics`.
- Abrir `http://localhost:5000` para MLflow.
- Abrir `http://localhost:9090` para Prometheus.
- Abrir `http://localhost:3000` para Grafana.
- Verificar el estado del workflow en GitHub Actions tras un push.
