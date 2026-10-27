"""Direct launcher for the standalone melodic-vocal dataset manager."""

from __future__ import annotations

import uvicorn

from autotransition.dataset_manager.app import create_app
from autotransition.dataset_manager.config import DatasetManagerConfig


if __name__ == "__main__":
    config = DatasetManagerConfig.from_env()
    uvicorn.run(create_app(config), host=config.host, port=config.port)
