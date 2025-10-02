# mlops-capstone-project
MLOPS Capstone Project

# Retail Store & Customer Insights

An end-to-end analytics pipeline & a FastAPI-powered web application for querying KPIs and serving insights to stakeholders.

## Run API

```bash
uvicorn src.api.app:create_app --factory --host 0.0.0.0 --port 8000
```

Open API docs at `http://localhost:8000/docs`.

### Endpoints
- `/health`
- `/metrics` (Prometheus)
- `/kpis/summary`
- `/customers/segments`
- `/customers/rfm`
- `/trends/seasonal`
- `/payments/analysis`
- `/profitability?discount_clothing=0.05&discount_shoes=0.03&discount_accessories=0.02`
- `/models/kmeans?n_clusters=5`
- `/forecast/monthly?periods=3`

## CI/CD & Monitoring
- GitHub Actions workflow at `.github/workflows/ci.yml` runs lint and tests on PRs and pushes.
- Prometheus metrics exposed at `/metrics`; scrape in your monitoring stack.
- For retraining, call `/models/kmeans?n_clusters=...` periodically or wire to a scheduler.