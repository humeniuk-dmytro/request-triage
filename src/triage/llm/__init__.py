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
from .ratelimit import RateLimitedClient, RateLimiter

__all__ = [
    "FatalLLMError",
    "LLMClient",
    "LLMError",
    "LLMResponse",
    "Message",
    "TransientLLMError",
    "FakeClient",
    "RateLimitedClient",
    "RateLimiter",
    "build_client",
]


def build_client(provider: str, settings: Settings) -> LLMClient:
    if provider == "fake":
        # Заглушка не ходить у мережу — обмежувати нічого.
        return FakeClient()
    if provider == "gemini":
        # Імпорт усередині: без ключа й без потреби в Gemini застосунок
        # має запускатись, навіть якщо SDK не встановлено.
        from .gemini import GeminiClient

        client: LLMClient = GeminiClient(api_key=settings.api_key or "", model=settings.model)
        if settings.rpm > 0:
            client = RateLimitedClient(client, settings.rpm)
        return client
    raise ValueError(f"Невідомий провайдер: {provider!r}. Доступні: gemini, fake.")
