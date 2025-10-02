from fastapi import FastAPI, Query, Request, Response
from typing import Optional
from ..data.processor import DataProcessor
import os
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST


def create_app(data_path: Optional[str] = None) -> FastAPI:
    data_path = data_path or os.path.join(os.path.dirname(__file__), "..", "data", "customer_shopping_data.csv")
    data_path = os.path.abspath(data_path)

    app = FastAPI(title="Retail Analytics API", version="1.0.0")

    # Initialize a single processor instance and store on app state
    processor = DataProcessor(data_path)
    processor.load_data()
    processor.clean_data()
    app.state.processor = processor

    # Prometheus metrics
    REQUEST_COUNT = Counter(
        'http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'http_status']
    )
    REQUEST_LATENCY = Histogram(
        'http_request_latency_seconds', 'Latency of HTTP requests in seconds', ['endpoint']
    )

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        start_time = time.time()
        response: Response = await call_next(request)
        latency = time.time() - start_time
        endpoint = request.url.path
        REQUEST_COUNT.labels(request.method, endpoint, str(response.status_code)).inc()
        REQUEST_LATENCY.labels(endpoint).observe(latency)
        return response

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/kpis/summary")
    def kpis_summary():
        summary = app.state.processor.get_data_summary()
        # Serialize date_range
        dr = summary["date_range"]
        summary["date_range"] = {"start": str(dr["start"]), "end": str(dr["end"]) }
        return summary

    @app.get("/customers/segments")
    def customer_segments():
        metrics = app.state.processor.calculate_customer_metrics()
        segmented = app.state.processor.segment_customers(metrics)
        return {
            "counts": segmented["customer_segment"].value_counts().to_dict(),
            "sample": segmented.head(10).reset_index().to_dict(orient="records"),
        }

    @app.get("/customers/rfm")
    def rfm_loyalty():
        return app.state.processor.rfm_loyalty_analysis()

    @app.get("/trends/seasonal")
    def seasonal_trends():
        return app.state.processor.analyze_seasonal_trends()

    @app.get("/payments/analysis")
    def payment_analysis():
        return app.state.processor.analyze_payment_methods()

    @app.get("/profitability")
    def profitability(
        discount_clothing: float = Query(0.05, ge=0.0, le=1.0),
        discount_shoes: float = Query(0.03, ge=0.0, le=1.0),
        discount_accessories: float = Query(0.02, ge=0.0, le=1.0),
    ):
        discount_map = {
            "Clothing": discount_clothing,
            "Shoes": discount_shoes,
            "Accessories": discount_accessories,
        }
        # Example category-level cost rates; can be adjusted or externalized
        cost_rate_map = {
            "Clothing": 0.55,
            "Shoes": 0.62,
            "Accessories": 0.48,
        }
        result = app.state.processor.calculate_profitability(
            discount_rate_by_category=discount_map,
            cost_rate_by_category=cost_rate_map,
        )
        return result

    @app.get("/models/kmeans")
    def train_kmeans(n_clusters: int = 5):
        return app.state.processor.train_kmeans_segmentation(n_clusters=n_clusters)

    @app.get("/forecast/monthly")
    def forecast_monthly(periods: int = 3):
        return app.state.processor.forecast_monthly_sales_naive(periods=periods)

    @app.get("/customers/top")
    def customers_top(top_percent: float = Query(0.10, ge=0.01, le=1.0), min_transactions: int = 1):
        return app.state.processor.top_customers(top_percent=top_percent, min_transactions=min_transactions)

    @app.get("/customers/repeat-vs-onetime")
    def repeat_vs_onetime():
        return app.state.processor.repeat_vs_onetime()

    @app.get("/kpis/categories")
    def categories_kpis():
        return app.state.processor.categories_kpis()

    @app.get("/simulation/campaign")
    def simulation_campaign(segment: str = 'High Value', discount: float = Query(0.10, ge=0.0, le=1.0)):
        return app.state.processor.simulate_campaign(segment=segment, discount=discount)

    return app


