"""Обмежувач частоти звернень.

Причина існування конкретна: безкоштовний тір Gemini дозволяє 5 запитів на
хвилину на модель. Семафор паралельності від цього не рятує — він обмежує,
скільки запитів летить *одночасно*, а ліміт рахує, скільки їх було *за
хвилину*. П'ять паралельних запитів проходять, наступні п'ять через секунду
вже ні.

Тому: рівномірне рознесення в часі (60/rpm між стартами) замість того, щоб
впертися в стіну й розгрібати 429 повторами. Повтор після відмови коштує
рівно стільки ж, скільки очікування до неї, тільки ще й додає навантаження.

Реалізовано декоратором над клієнтом, а не всередині конвеєра: так під ліміт
потрапляють усі звернення, включно з repair-викликами й проходом по дублях.
"""

from __future__ import annotations

import asyncio
from typing import Any

from .base import LLMClient, LLMResponse, Message


class RateLimiter:
    """Не частіше ніж `rpm` стартів на хвилину, рівномірно."""

    def __init__(self, rpm: float) -> None:
        self._interval = 60.0 / rpm if rpm > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next_slot = 0.0

    async def acquire(self) -> None:
        if self._interval <= 0:
            return
        # Слот бронюється під замком, а чекається вже без нього: інакше
        # паралельні задачі стояли б у черзі за самим замком.
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self._interval
        delay = slot - asyncio.get_running_loop().time()
        if delay > 0:
            await asyncio.sleep(delay)


class RateLimitedClient:
    """Прозора обгортка: той самий інтерфейс, але з дотриманням ліміту."""

    def __init__(self, inner: LLMClient, rpm: float) -> None:
        self._inner = inner
        self._limiter = RateLimiter(rpm)
        self.name = inner.name
        self.model = inner.model

    async def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        schema: dict[str, Any],
    ) -> LLMResponse:
        await self._limiter.acquire()
        return await self._inner.complete(system=system, messages=messages, schema=schema)

    async def aclose(self) -> None:
        await self._inner.aclose()
