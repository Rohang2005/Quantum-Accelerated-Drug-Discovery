"""Print every Gemini model that supports ``generateContent`` for the API key in .env."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import google.generativeai as genai

from src.config import require_google_api_key


def main() -> None:
    genai.configure(api_key=require_google_api_key())
    print("Models supporting generateContent:")
    for model in genai.list_models():
        if "generateContent" in model.supported_generation_methods:
            print(f"  - {model.name}")


if __name__ == "__main__":
    main()
