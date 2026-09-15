"""YuE2-3B native audio.cpp worker."""

from .config import Yue2Config
from .runtime import Yue2Runtime

__all__ = ["Yue2Config", "Yue2Runtime", "create_yue2_worker_app"]


def create_yue2_worker_app(*args, **kwargs):
    """Load the server factory lazily so ``python -m ...server`` initializes once."""
    from .server import create_yue2_worker_app as factory

    return factory(*args, **kwargs)
