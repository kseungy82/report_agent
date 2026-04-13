from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    upstage_api_key: str | None = os.getenv("UPSTAGE_API_KEY") or None
    huggingfacehub_api_token: str | None = os.getenv("HUGGINGFACEHUB_API_TOKEN") or None
    kanana_model_id: str = os.getenv("KANANA_MODEL_ID", "kakaocorp/kanana-1.5-2.1b-instruct-2505")
    kanana_model_path: str | None = os.getenv("KANANA_MODEL_PATH") or None 
    app_port: int = int(os.getenv("APP_PORT", "8000"))
    log_dir: str = os.getenv("LOG_DIR", ".")
    runtime_dir: str = os.getenv("RUNTIME_DIR", "runtime_files")


settings = Settings()
