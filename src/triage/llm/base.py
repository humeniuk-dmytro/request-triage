"""Контракт LLM-клієнта.

Розділення помилок на два класи тут — не формальність. Транспортний збій
(429, 503, таймаут) лікується повтором того самого запиту: він випадковий.
Невалідний вивід повтором не лікується — модель віддасть те саме; його треба
або нормалізувати, або показати моделі помилку. Тому й обробники різні.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

Role = Literal["user", "model"]
Message = tuple[Role, str]


class LLMError(Exception):
    """Базова помилка звернення до моделі."""


class TransientLLMError(LLMError):
    """Тимчасовий збій: rate limit, 5xx, таймаут, обрив. Має сенс повторити.

    `retry_after` — витримка, яку назвав сам сервер. Коли Gemini у відповіді на
    429 пише «Please retry in 28s», вгадувати експонентою безглуздо: наша
    здогадка в 4 секунди — це просто ще один 429.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class FatalLLMError(LLMError):
    """Постійний збій: невірний ключ, немає доступу до моделі, 400. Повтор не допоможе."""


@dataclass(slots=True)
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LLMClient(Protocol):
    """Мінімум, якого достатньо і для класифікації, і для проходу по дублях.

    Свідомо вузький: додати ще одного провайдера — це один файл із одним
    методом, без переписування конвеєра.
    """

    name: str
    model: str

    async def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        schema: dict[str, Any],
    ) -> LLMResponse:
        """Повертає сирий текст відповіді. Валідація — не турбота клієнта."""
        ...

    async def aclose(self) -> None: ...
