"""key.env 로딩 + 벽 사진 캡션(OpenRouter).

README 규칙: OpenRouter 키는 `key.env` 파일로 따로 전달되며 **하드코딩하지
말고 그 파일에서 읽어야** 한다. 이 모듈이 그 로딩과 호출을 담당한다.

## 왜 백그라운드 스레드인가

QA 단계가 시작되면 메모리는 **동결**된다(README: 평가기는 에피소드가 끝나면
act를 더 부르지 않는다). 그러니 캡션은 반드시 play 단계의 `act` 안에서
붙여야 한다. 그런데 `act`의 예산은 ~5초이고 답변은 10초다. VLM 호출 한 번이
1~3초씩 걸리므로 **`act` 안에서 동기로 부르면 예산이 바로 무너진다.**

그래서 캡션 요청은 큐에 넣고 워커 스레드가 처리한다. `act`는 큐에 넣는
비용(마이크로초)만 부담하고 즉시 돌아온다. 결과는 Photo 객체에 나중에
채워진다 — 에피소드가 10분이라 대부분의 사진은 QA 전에 캡션이 붙는다.

## 비용

크레딧은 200달러 고정이고 시드마다 사진이 30~40장이다. 그래서
`MAX_CALLS`로 에피소드당 호출 수를 못 박고, 큐도 유한하게 둔다. 예산을
넘기면 조용히 크롭+서명만 남긴다(그 자체로 "같은 카테고리 사진 두 장" 류의
유사도 질문은 답할 수 있다).

## 키가 없을 때

키가 없거나 네트워크가 막혀 있으면 **조용히 비활성화**된다. 캡션이 빈
문자열로 남을 뿐 에이전트는 정상 동작한다. 개발 환경에 key.env가 없는 게
정상이므로 예외를 던지지 않는다.
"""

from __future__ import annotations

import base64
import io
import json
import os
import queue
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

# key.env에서 찾아볼 이름들. 어떤 이름으로 오든 받아들인다.
_KEY_NAMES = ("OPENROUTER_API_KEY", "OPENROUTER_KEY", "OPENROUTER_API_TOKEN",
              "API_KEY", "KEY")
_MODEL_NAMES = ("OPENROUTER_MODEL", "MODEL", "VLM_MODEL")

# key.env를 찾아볼 위치. 평가 하네스가 어디서 실행할지 모르므로 넉넉히 본다.
_SEARCH = (
    Path.cwd() / "key.env",
    Path(__file__).resolve().parent / "key.env",
    Path(__file__).resolve().parent.parent / "key.env",
    Path.home() / "key.env",
)

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
# 모델 id는 **설정으로 뺀다**. OpenRouter의 카탈로그는 수시로 바뀌고 이
# 개발 환경에서는 확인할 방법이 없다. key.env에 MODEL=... 을 넣으면 그걸
# 쓰고, 없으면 아래 기본값을 쓴다. 기본값이 그 계정에서 안 먹으면 캡션만
# 조용히 비게 되고(에이전트는 정상 동작) stats에 실패 수가 남는다.
DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"

MAX_CALLS = 60          # 에피소드당 캡션 호출 상한(크레딧 보호)
MAX_QUEUE = 40          # 대기 큐 상한. 넘치면 새 요청을 버린다.
TIMEOUT_S = 20.0
PROMPT = ("Name the single main subject of this photo in one or two words, "
          "lowercase, no punctuation. Answer with the noun only.")


@dataclass
class VLMConfig:
    api_key: str = ""
    model: str = DEFAULT_MODEL
    source: str = ""            # 어디서 읽었는지(진단용)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


def _parse_env(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def load_config(path: Optional[Path] = None) -> VLMConfig:
    """key.env(또는 환경변수)에서 키와 모델을 읽는다. 없으면 비활성 설정."""
    candidates: List[Path] = [path] if path else list(_SEARCH)
    env_path = os.environ.get("KEY_ENV")
    if env_path:
        candidates.insert(0, Path(env_path))
    for p in candidates:
        try:
            if p and p.is_file():
                data = _parse_env(p.read_text(encoding="utf-8"))
                key = next((data[n] for n in _KEY_NAMES if data.get(n)), "")
                if key:
                    model = next((data[n] for n in _MODEL_NAMES if data.get(n)),
                                 DEFAULT_MODEL)
                    return VLMConfig(api_key=key, model=model, source=str(p))
        except OSError:
            continue
    # 파일이 없으면 환경변수도 본다.
    key = next((os.environ[n] for n in _KEY_NAMES if os.environ.get(n)), "")
    if key:
        model = next((os.environ[n] for n in _MODEL_NAMES if os.environ.get(n)),
                     DEFAULT_MODEL)
        return VLMConfig(api_key=key, model=model, source="env")
    return VLMConfig()


def _png_data_url(rgb) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def caption_once(cfg: VLMConfig, rgb, timeout: float = TIMEOUT_S) -> str:
    """사진 한 장에 캡션을 붙인다(동기). 실패하면 빈 문자열.

    OpenAI 호환 chat/completions 형식이라 표준 라이브러리만으로 충분하다
    (이 venv에는 requests/httpx가 없고, 에이전트에 의존성을 늘리고 싶지 않다).
    """
    if not cfg.enabled:
        return ""
    body = json.dumps({
        "model": cfg.model,
        "max_tokens": 24,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": _png_data_url(rgb)}},
        ]}],
    }).encode()
    req = urllib.request.Request(
        ENDPOINT, data=body, method="POST",
        headers={"Authorization": f"Bearer {cfg.api_key}",
                 "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
        text = payload["choices"][0]["message"]["content"]
        return " ".join(str(text).split()).strip().strip(".").lower()[:40]
    except (urllib.error.URLError, KeyError, IndexError, ValueError,
            TimeoutError, OSError):
        return ""


class Captioner:
    """캡션을 백그라운드로 붙인다. `act`는 절대 블로킹되지 않는다.

    사용법:
        cap = Captioner()          # 키가 없으면 비활성 상태로 조용히 산다
        cap.submit(photo)          # 즉시 반환
        ...                        # photo.caption이 나중에 채워진다
    """

    def __init__(self, cfg: Optional[VLMConfig] = None,
                 max_calls: int = MAX_CALLS):
        self.cfg = cfg or load_config()
        self.max_calls = max_calls
        self.sent = 0
        self.done = 0
        self.failed = 0
        self.dropped = 0
        self._q: "queue.Queue" = queue.Queue(maxsize=MAX_QUEUE)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled

    def submit(self, photo, on_done: Optional[Callable] = None) -> bool:
        """캡션 요청을 큐에 넣는다. 넣었으면 True. 절대 블로킹하지 않는다."""
        if not self.enabled or self.sent >= self.max_calls:
            return False
        try:
            self._q.put_nowait((photo, on_done))
        except queue.Full:
            self.dropped += 1
            return False
        self.sent += 1
        self._ensure_worker()
        return True

    def _ensure_worker(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                photo, on_done = self._q.get(timeout=1.0)
            except queue.Empty:
                return                      # 할 일이 없으면 스레드를 접는다
            text = caption_once(self.cfg, photo.crop)
            if text:
                photo.caption = text
                self.done += 1
            else:
                self.failed += 1
            if on_done is not None:
                try:
                    on_done(photo)
                except Exception:           # noqa: BLE001 - 콜백이 죽어도 계속
                    pass
            self._q.task_done()

    def stats(self) -> dict:
        return {"enabled": self.enabled, "model": self.cfg.model,
                "source": self.cfg.source, "sent": self.sent,
                "captioned": self.done, "failed": self.failed,
                "dropped": self.dropped}

    def shutdown(self) -> None:
        self._stop.set()
