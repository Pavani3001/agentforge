"""Small OpenAI-compatible HTTP client used by the AgentForge pipeline."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable
from urllib import error, request


class LLMError(RuntimeError):
    """Raised when an LLM request cannot be completed or decoded."""


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    timeout: float = 60.0

    @classmethod
    def from_env(cls) -> "LLMConfig":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise LLMError(
                "OPENAI_API_KEY is not configured. Set it before running the pipeline."
            )
        return cls(
            api_key=api_key,
            base_url=os.getenv(
                "OPENAI_BASE_URL", "https://api.openai.com/v1"
            ).rstrip("/"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            timeout=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "60")),
        )


class OpenAICompatibleClient:
    """Client for chat-completions compatible APIs, including OpenAI proxies."""

    def __init__(
        self,
        config: LLMConfig | None = None,
        opener: Callable[..., Any] = request.urlopen,
    ) -> None:
        self.config = config or LLMConfig.from_env()
        self._opener = opener

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        payload = {
            "model": self.config.model,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        endpoint = f"{self.config.base_url}/chat/completions"
        req = request.Request(
            endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with self._opener(req, timeout=self.config.timeout) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMError(f"LLM request failed ({exc.code}): {detail[:500]}") from exc
        except (error.URLError, TimeoutError, OSError) as exc:
            raise LLMError(f"LLM request could not be completed: {exc}") from exc

        try:
            envelope = json.loads(raw)
            content = envelope["choices"][0]["message"]["content"]
            result = json.loads(content)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError("LLM returned an invalid JSON response.") from exc
        if not isinstance(result, dict):
            raise LLMError("LLM JSON response must be an object.")
        return result
