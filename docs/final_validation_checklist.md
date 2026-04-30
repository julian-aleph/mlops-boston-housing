# Final validation checklist

Lista de comandos y checks manuales para validar el entregable antes de la
presentacion.

## Comandos automatizables

```bash
make setup          # instalar dependencias en .venv activo
make pipeline       # data + features + train + evaluate + promote
make api-check      # importar app.main con el modelo promovido
make serve          # levantar uvicorn local en :8000 (Ctrl-C para terminar)
make docker-build   # construir boston-housing-api:local
make docker-up      # levantar api + prometheus + grafana
make monitor        # imprimir URLs de monitoreo
make lint           # ruff + compileall
make test           # pytest
dvc status
dvc metrics show
```

## Checks manuales

- [ ] Abrir FastAPI docs en `http://localhost:8000/docs` y disparar `/predict`
      desde la UI Swagger.
- [ ] Probar `/predict` con `curl` (payload del README seccion 9).
- [ ] Verificar `/health` retorna `status=ok` y `model_loaded=true`.
- [ ] Verificar `/metrics` retorna formato Prometheus con
      `prediction_requests_total`.
- [ ] Abrir MLflow UI en `http://localhost:5000` (`make mlflow-ui`) y revisar
      runs por candidato.
- [ ] Abrir Prometheus en `http://localhost:9090` y verificar que el target
      `boston-housing-api` aparece UP.
- [ ] Abrir Grafana en `http://localhost:3000` (admin / admin) y abrir el
      dashboard provisionado "Boston Housing API".
- [ ] Verificar status verde del workflow en GitHub Actions
      (`.github/workflows/ci.yml`).
- [ ] Apagar el stack con `make docker-down`.

## Senales de completitud

- `make pipeline` regenera `models/production/{model.joblib, metadata.json}`.
- `make test` reporta verde (algunos tests del API hacen skip si falta
  `httpx`; instalado via `requirements.txt`).
- `make lint` reporta `All checks passed!`.
- `make docker-build` produce la imagen `boston-housing-api:local`.
- `dvc status` reporta `Data and pipelines are up to date.` despues de
  ejecutar `dvc repro`.
