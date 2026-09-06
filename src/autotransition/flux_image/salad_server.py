"""Production HTTP process for the FLUX image Salad queue transport."""

from __future__ import annotations

import os

import uvicorn

from .config import FluxImageConfig
from .worker import create_flux_image_worker_app


config = FluxImageConfig.from_env()
app = create_flux_image_worker_app(config)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("WORKER_HOST", "0.0.0.0"),
        port=int(os.getenv("WORKER_PORT", "8080")),
        log_level=os.getenv("UVICORN_LOG_LEVEL", "info"),
    )
