"""Config: OpenRouter API key loading, model/threshold constants.

The key is never hardcoded. Lookup order:
  1) OPENROUTER_API_KEY environment variable
  2) Glob for key*.env in the project root (this file's grandparent dir)
     -- supports both "OPENROUTER_API_KEY=..." format and a bare key on
     one line (that's the format we actually got during development).
The README only says "key.env" without pinning down the exact filename,
and evaluation may use a different one, so we glob instead of hardcoding
a single name.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

VLM_MODEL = "google/gemini-3.1-flash-lite"
VLM_TIMEOUT_S = 4.0
VLM_MAX_CALLS_PER_EPISODE = 60  # safety cap on credit/time spend

# QA fallback is text-only (frozen JSON + question, no images), so a
# cheap chat model is enough. DeepSeek-R1 / gpt-5-nano spend the 10 s
# answer budget on hidden reasoning tokens; V4 Flash does not.
QA_FALLBACK_MODEL = "deepseek/deepseek-v4-flash"
QA_TIMEOUT_S = 4.0

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_key_file(path: Path) -> Optional[str]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    if "=" in text.splitlines()[0]:
        # "OPENROUTER_API_KEY=sk-..." format.
        return text.splitlines()[0].split("=", 1)[1].strip()
    return text  # bare key, one line.


def get_api_key() -> Optional[str]:
    env_key = os.environ.get("OPENROUTER_API_KEY")
    if env_key:
        return env_key.strip()
    for candidate in _PROJECT_ROOT.glob("key*.env"):
        try:
            key = _parse_key_file(candidate)
            if key:
                return key
        except OSError:
            continue
    return None
