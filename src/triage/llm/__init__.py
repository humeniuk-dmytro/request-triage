"""Вибір провайдера. Додати нового — це новий файл із одним методом complete()."""

from __future__ import annotations

from ..config import Settings
from .base import (
    FatalLLMError,
    LLMClient,
    LLMError,
    LLMResponse,
    Message,
    TransientLLMError,
)
from .fake import FakeClient

__all__ = [
    "FatalLLMError",
    "LLMClient",
    "LLMError",
    "LLMResponse",
    "Message",
    "TransientLLMError",
    "FakeClient",
    "build_client",
]


def build_client(provider: str, settings: Settings) -> LLMClient:
    if provider == "fake":
        return FakeClient()
    if provider == "gemini":
        # Імпорт усередині: без ключа й без потреби в Gemini застосунок
        # має запускатись, навіть якщо SDK не встановлено.
        from .gemini import GeminiClient

        return GeminiClient(api_key=settings.api_key or "", model=settings.model)
    raise ValueError(f"Невідомий провайдер: {provider!r}. Доступні: gemini, fake.")
