"""Production HTTP process for the Salad queue transport."""

from __future__ import annotations

import os

import uvicorn

from autotransition.config import AvatarConfig

from .salad_adapter import install_salad_routes
from .worker import create_avatar_worker_app


config = AvatarConfig.from_env()
app = create_avatar_worker_app(config)
install_salad_routes(app, app.state.avatar_worker, config)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("WORKER_HOST", "0.0.0.0"),
        port=int(os.getenv("WORKER_PORT", "8080")),
        log_level=os.getenv("UVICORN_LOG_LEVEL", "info"),
    )
