# Presentation outline (20 minutes)

Guion para una presentacion de 20 minutos. Cada seccion incluye mensaje clave,
2 a 4 puntos para apoyar y que enfatizar verbalmente.

## 1. Problem and constraints (1.5 min)

**Mensaje clave:** entregable MLOps end-to-end para regresion sobre Boston
Housing, alineado a 6–8 horas, local-first y open-source.

- Problema: predecir `MEDV` y exponerlo como REST API.
- Restricciones: sin cloud, sin K8s, sin servicios administrados, sin secrets.
- Foco: reproducibilidad, separacion de responsabilidades y observabilidad
  basica.

**Enfasis verbal:** las decisiones se toman pensando en migracion futura a
cloud sin reescribir codigo.

## 2. Architecture overview (2 min)

**Mensaje clave:** stack open-source que cubre todo el ciclo de vida del
modelo en una sola maquina.

- Pipeline: data → features → train → evaluate → promote.
- Artefactos versionados con DVC; experimentos rastreados con MLflow.
- Serving con FastAPI; despliegue local con Docker Compose.
- Monitoreo con Prometheus + Grafana; CI con GitHub Actions.

**Enfasis verbal:** cada herramienta se eligio porque es portable y se reemplaza
limpiamente al pasar a cloud.

## 3. Data preparation and feature management (2 min)

**Mensaje clave:** datos validados, perfilados y registrados antes de entrenar.

- Descarga reproducible desde KaggleHub con cache local de fallback.
- Validacion de columnas requeridas y conteo minimo de filas.
- Perfil minimo de datos en `reports/metrics/data_profile.json`.
- Feature registry con schema y version en `data/processed/`.

**Enfasis verbal:** cualquier cambio de schema rompe el contrato y se detecta
temprano.

## 4. Training and model selection (2 min)

**Mensaje clave:** seleccion de modelo basada en `RandomizedSearchCV` con RMSE
como metrica primaria.

- Candidatos: Ridge, ElasticNet, RandomForest, HistGradientBoosting.
- `sklearn Pipeline` empaqueta preprocesamiento + estimador.
- `QuantileClipper` custom para tratamiento de outliers consistente entre
  train e inference.
- Promocion a produccion solo si supera el umbral de mejora de RMSE.

**Enfasis verbal:** el preprocesamiento viaja con el modelo, evitando skew
entre training y serving.

## 5. MLflow and DVC (2 min)

**Mensaje clave:** trazabilidad de experimentos y reproducibilidad del DAG con
herramientas independientes pero complementarias.

- MLflow: SQLite backend + artifact store local; un run por candidato.
- DVC: `dvc.yaml` con stages para cada paso; `dvc.lock` reproducible.
- MLflow no reemplaza joblib; el serving consume `models/production/`.
- Sin remote cloud configurado; trivial de agregar.

**Enfasis verbal:** se puede reproducir el pipeline desde cero con
`dvc repro` o `make pipeline`.

## 6. FastAPI serving (2 min)

**Mensaje clave:** API minimal que carga el artefacto promovido y valida payloads.

- Endpoints `/predict`, `/health`, `/metrics`.
- Schemas Pydantic con `extra="forbid"` rechazan campos desconocidos.
- Loader unico (`app/model_loader.py`) con paths configurables por env vars.
- API no depende de MLflow ni DVC en runtime.

**Enfasis verbal:** la API es desplegable de forma independiente del stack de
training.

## 7. Docker deployment (2 min)

**Mensaje clave:** imagen serving aislada y reproducible.

- Base `python:3.11-slim`, solo dependencias necesarias.
- Copia `app/`, `src/` (para `QuantileClipper`), `configs/` y
  `models/production/`.
- `docker-compose.yml` monta modelo y configs read-only.
- Training queda explicitamente fuera de la imagen.

**Enfasis verbal:** entrenar y servir son ciclos de vida separados; reentrenar
no requiere rebuild.

## 8. Monitoring (2 min)

**Mensaje clave:** observabilidad operativa basica con metricas estandar.

- `prediction_requests_total`, `prediction_errors_total`,
  `prediction_latency_seconds`, `prediction_value`, `model_loaded`,
  `model_info`.
- Prometheus scrape cada 15s al servicio `api`.
- Grafana provisionado con datasource y dashboard listos.

**Enfasis verbal:** drift avanzado queda como mejora; el setup actual cubre
salud del servicio y distribucion de predicciones.

## 9. CI/CD and quality checks (1.5 min)

**Mensaje clave:** validacion automatica en cada push y PR.

- Ruff + compileall para estilo y bytecode.
- Pytest para contratos de API, training y preprocesamiento.
- Pipeline end-to-end ejecutado en CI.
- Build de imagen Docker como senal de readiness.

**Enfasis verbal:** el workflow se mantiene en un solo job para evitar
complejidad innecesaria.

## 10. Strategic roadmap and trade-offs (3 min)

**Mensaje clave:** lo que esta intencionalmente fuera del alcance y como
crecer desde aqui.

- Hardening: auth, rate limiting, secrets management.
- Cloud: Terraform, S3/GCS para DVC remote, Postgres para MLflow.
- Governance: MLflow Model Registry, drift con Evidently/WhyLabs.
- Despliegue: canary/blue-green, registry de imagenes, autoscaling.

**Enfasis verbal:** el stack actual es la base mas pequena que demuestra
todas las capacidades sin acoplarse a un proveedor.
