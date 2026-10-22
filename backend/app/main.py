from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .api.routes import router
import logging
import time
import uuid
from fastapi import Request
from .logging_config import configure_logging
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from .rate_limit import limiter

configure_logging()
logger = logging.getLogger("cost_aware_rag.request")
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Backend API for Cost-Aware RAG with Semantic Caching and Intelligent Model Routing.",
)
app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler,
)

# Enable CORS for local Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
)

@app.middleware("http")
async def log_http_request(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception as exc:
        latency_ms = round(
            (time.perf_counter() - start_time) * 1000, 2
        )
        logger.error(
            "http_request_failed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": 500,
                "latency_ms": latency_ms,
                "error_type": type(exc).__name__,
            },
        )
        raise

    latency_ms = round(
        (time.perf_counter() - start_time) * 1000, 2
    )
    response.headers["X-Request-ID"] = request_id

    log_method = logger.warning if response.status_code >= 400 else logger.info
    log_method(
        "http_request_completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
        },
    )

    return response


app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
