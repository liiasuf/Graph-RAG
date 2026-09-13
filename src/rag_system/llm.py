from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from openai import APIConnectionError, APITimeoutError, OpenAI


@dataclass(frozen=True)
class LLMUsage:
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


@dataclass(frozen=True)
class LLMResult:
    text: str
    usage: LLMUsage
    raw: Any


@dataclass
class RemoteLLM:
    """
    Chat client for OpenAI-API-compatible HTTP servers.
    Use base_url and model to switch between different providers and models.
    """

    base_url: str
    api_key: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 128
    timeout_s: float = 120.0

    def _client(self):
        return OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout_s)

    def generate(self, messages, _retries: int = 4):
        """
        Retries on transient connection drops/timeouts (e.g. a VPN blip
        during a multi-hour sweep) with increasing backoff (3s/6s/12s),
        so a few seconds of lost connectivity doesn't abort the whole run.
        Does NOT retry BadRequestError (4xx) - that's a real client error
        the caller (pipeline.run_eval_loop) already has its own fallback
        for, and retrying it would just waste time reproducing the same
        error.
        """
        last_exc: Exception | None = None
        for attempt in range(_retries):
            try:
                client = self._client()
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                text = (resp.choices[0].message.content or "").strip()

                usage = getattr(resp, "usage", None)

                u = LLMUsage(
                    prompt_tokens=getattr(usage, "prompt_tokens", None) if usage is not None else None,
                    completion_tokens=(
                        getattr(usage, "completion_tokens", None) if usage is not None else None
                    ),
                    total_tokens=getattr(usage, "total_tokens", None) if usage is not None else None,
                )

                return LLMResult(text=text, usage=u, raw=resp)
            except (APIConnectionError, APITimeoutError) as e:
                last_exc = e
                if attempt < _retries - 1:
                    time.sleep(min(3 * (2**attempt), 30))
                else:
                    raise
        raise last_exc  # pragma: no cover - unreachable, loop always returns or raises
