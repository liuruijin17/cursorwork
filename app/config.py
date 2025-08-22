import os
from typing import Optional


def get_env(name: str, default: Optional[str] = None) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        return default if default is not None else ""
    return value


UPSTREAM_BASE_URL: str = get_env("UPSTREAM_BASE_URL", "http://127.0.0.1:8000/v1")
TOKEN_LOG_PATH: str = get_env("TOKEN_LOG_PATH", "/workspace/token_logs.jsonl")
UPSTREAM_TIMEOUT_SECS: float = float(get_env("UPSTREAM_TIMEOUT_SECS", "300"))
LOG_ROTATE_MAX_BYTES: int = int(get_env("LOG_ROTATE_MAX_BYTES", str(100 * 1024 * 1024)))
LOG_ROTATE_KEEP: int = int(get_env("LOG_ROTATE_KEEP", "5"))