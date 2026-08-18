"""설정: OpenRouter API 키 로딩, 모델/임계값 상수.

키는 코드에 하드코딩하지 않는다. 우선순위:
  1) 환경변수 OPENROUTER_API_KEY
  2) 프로젝트 루트(이 파일의 조부모 디렉터리)에서 key*.env 글롭 검색
     - 파일 내용이 "OPENROUTER_API_KEY=..." 형식이든, 그냥 키 원문
       한 줄이든(개발 중 실제로 이 형식이었음 — dev_log.md 참고) 둘 다 지원.
평가 환경에서 정확한 파일명을 모르므로(README는 그냥 "key.env"라고만
지칭) 글롭으로 유연하게 찾는다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

VLM_MODEL = "google/gemini-3.1-flash-lite"
VLM_TIMEOUT_S = 4.0
VLM_MAX_CALLS_PER_EPISODE = 60  # 크레딧/시간 안전판

QA_FALLBACK_MODEL = "openai/gpt-5-nano"
QA_TIMEOUT_S = 4.0

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_key_file(path: Path) -> Optional[str]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    if "=" in text.splitlines()[0]:
        # "OPENROUTER_API_KEY=sk-..." 형식.
        return text.splitlines()[0].split("=", 1)[1].strip()
    return text  # 원문 키 한 줄.


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
