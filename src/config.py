"""Configuration loader.

Reads ``.env`` (for the Google API key) and ``config.json`` (for everything
tunable about the pipeline) and exposes them as plain dictionaries.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        with env_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.json"
DATA_DIR = PROJECT_ROOT / "data"


def _load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


_config = _load_config()

LLM_CONFIG: Dict[str, Any] = _config.get("llm", {})
SIMULATION_CONFIG: Dict[str, Any] = _config.get("simulation", {})
GENERATION_CONFIG: Dict[str, Any] = _config.get("generation", {})
QUANTUM_CONFIG: Dict[str, Any] = _config.get("quantum", {})
MOLGAN_CONFIG: Dict[str, Any] = _config.get("molgan", {})


def require_google_api_key() -> str:
    """Return the Google API key or raise if it is missing.

    We resolve this lazily so importing :mod:`src.config` does not crash users
    who only want to use the quantum or GAN modules without an LLM key.
    """
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError(
            "GOOGLE_API_KEY is not set. Add it to .env or export it before "
            "running the LLM-dependent steps."
        )
    return key
