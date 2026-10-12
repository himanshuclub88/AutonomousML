from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

# Load a project-local .env first, then allow an environment-provided .env.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(r"G:\Other computers\My Laptop\Study\.env")
load_dotenv()

@dataclass(frozen=True)
class Settings:
    llm_api_key: str
    llm_base_url: str
    llm_model: str
    timezone: str

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

def _get(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()

settings = Settings(
    llm_api_key=_get("LLM_API_KEY"),
    llm_base_url=_get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
    llm_model=_get("LLM_MODEL"),
    timezone=_get("TIMEZONE", "Asia/Kolkata"),
)

COLLAB_RUNTIME = _get('COLLAB_RUNTIME')
