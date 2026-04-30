# MLOps local - Boston Housing

Proyecto minimo y local para preparar datos, entrenar modelos de regresion y
persistir un modelo candidato para Boston Housing.

## Alcance implementado

- Descarga del dataset desde KaggleHub.
- Validacion del dataset raw.
- Perfil minimo de datos raw.
- Preparacion de features.
- Preprocesamiento reutilizable con `QuantileClipper`.
- Entrenamiento con `sklearn Pipeline`.
- Evaluacion en hold-out reproducible.
- Importancia de features por permutacion.
- Persistencia de modelo en staging.
- Promocion local de staging a produccion.
- API local con FastAPI para servir el modelo en produccion.
- Tracking de experimentos con MLflow durante entrenamiento.
- Versionado de artefactos y DAG reproducible con DVC.
- Despliegue local del API con Docker y Docker Compose.

Monitoreo y GitHub Actions se agregaran en fases posteriores.

## Comandos

```bash
make setup
make data
make features
make train
make evaluate
make promote
make pipeline
make retrain
make serve
make api-check
make mlflow-ui
make dvc-repro
make dvc-status
make dvc-metrics
make test
make lint
```

## Artefactos generados

- `data/raw/boston_housing.csv`
- `reports/metrics/data_profile.json`
- `data/processed/boston_housing_processed.parquet`
- `data/processed/feature_schema.json`
- `data/processed/feature_registry.json`
- `models/staging/model.joblib`
- `models/production/model.joblib`
- `models/production/metadata.json`
- `models/production/metrics.json`
- `reports/metrics/staging_metrics.json`
- `reports/metrics/model_comparison.csv`
- `reports/feature_importance/feature_importance.csv`
- `reports/metrics/promotion_report.json`

## Modelos candidatos

- Ridge
- ElasticNet
- RandomForestRegressor
- HistGradientBoostingRegressor

## Uso de sklearn Pipeline

Cada modelo se entrena dentro de un `sklearn Pipeline` para mantener consistente
el preprocesamiento durante entrenamiento e inferencia. Esto reduce diferencias
entre entrenamiento y uso posterior del modelo, y deja el artefacto persistido
con el preprocesamiento necesario incluido.

## MLflow tracking

MLflow forma parte del stage de entrenamiento. `make train` registra un run por
modelo candidato con hiperparametros, metricas y el tag de modelo seleccionado.

MLflow usa SQLite como backend store local en `mlflow.db` y guarda artifacts en
`mlruns/`. Esto evita limitaciones del tracking basado solo en archivos dentro
de la UI de MLflow, sin requerir un servidor remoto.

MLflow no reemplaza la persistencia local con joblib: `models/staging/model.joblib`
sigue siendo el artefacto usado por promocion y serving local. La API FastAPI no
depende de MLflow.

```bash
make train
make mlflow-ui
```

```text
http://localhost:5000
```

## Artifact versioning and pipeline reproducibility with DVC

DVC formaliza el pipeline como un DAG y versiona artefactos generados sin
commitear outputs pesados directamente en Git. El Makefile sigue siendo la
interfaz operativa diaria; DVC se usa cuando se necesita reproducibilidad,
seguimiento de dependencias o versionado de artefactos.

No hay remote cloud configurado para este challenge. Una mejora futura podria
agregar un remote local, S3, GCS o SSH.

```bash
make dvc-repro
make dvc-status
make dvc-metrics
```

Equivalente directo:

```bash
dvc repro
dvc status
dvc metrics show
```

## API serving

Ensure a production model exists:

```bash
make pipeline
```

Start the API:

```bash
make serve
```

Check API import:

```bash
make api-check
```

Open the interactive docs:

```text
http://localhost:8000/docs
```

Example prediction request:

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

## Local deployment with Docker

La imagen Docker es solo para serving. El entrenamiento se ejecuta fuera de la
imagen mediante el Makefile o el pipeline de DVC. El contenedor carga el
artefacto de produccion desde `models/production`, lo que mantiene separadas
las responsabilidades de entrenamiento y serving.

Asegurar que existe un modelo de produccion antes de construir la imagen:

```bash
make pipeline
```

Construir la imagen del API:

```bash
make docker-build
```

Levantar el API con Docker Compose:

```bash
make docker-up
```

Ver logs del contenedor:

```bash
make docker-logs
```

Detener servicios:

```bash
make docker-down
```

Endpoints disponibles:

```text
http://localhost:8000/docs
http://localhost:8000/health
```

Ejemplo de prediccion contra el contenedor:

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
