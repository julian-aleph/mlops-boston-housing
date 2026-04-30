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

API, Docker, monitoreo, MLflow, DVC y GitHub Actions se agregaran en fases
posteriores.

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
