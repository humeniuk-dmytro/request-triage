"""Клієнт Gemini (безкоштовний тір AI Studio).

Обмеження виводу тут дворівневе ще до нашої валідації:
`response_mime_type="application/json"` прибирає прозу навколо JSON, а
`response_schema` не дає моделі вигадати поля чи значення enum. Це не робить
валідацію зайвою — це робить її рідше потрібною, а отже дешевшою.
"""

from __future__ import annotations

import re
from typing import Any

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from .base import FatalLLMError, LLMResponse, Message, TransientLLMError

# Коди, які лікуються повтором того самого запиту.
_RETRYABLE_CODES = {408, 409, 429, 500, 502, 503, 504}

# Gemini називає рекомендовану витримку у двох виглядах: як RetryInfo
# ('retryDelay': '28s') і як текст у message ('Please retry in 28.4s').
_RETRY_DELAY_RE = re.compile(
    r"retryDelay['\"]?\s*:\s*['\"](\d+(?:\.\d+)?)s|retry in (\d+(?:\.\d+)?)s"
)

# Стеля на витримку від сервера: при вичерпаній добовій квоті він може
# попросити чекати годинами — це вже не повтор, а зупинка роботи.
_MAX_SERVER_DELAY_SEC = 90.0


def _retry_after(exc: Exception) -> float | None:
    """Витримка, яку назвав сам сервер. Вона точніша за будь-яку нашу здогадку."""
    match = _RETRY_DELAY_RE.search(str(exc))
    if not match:
        return None
    return min(float(match.group(1) or match.group(2)), _MAX_SERVER_DELAY_SEC)


def _short(exc: Exception) -> str:
    """Помилки Gemini приїжджають портянкою на пів екрана — лишаємо суть."""
    text = " ".join(str(exc).split())
    match = re.search(r"'message':\s*'([^']{0,200})", text)
    return match.group(1) if match else text[:200]


class GeminiClient:
    name = "gemini"

    def __init__(self, api_key: str, model: str, timeout_sec: float = 60.0) -> None:
        if not api_key:
            raise FatalLLMError(
                "Немає GEMINI_API_KEY. Візьміть безкоштовний ключ на "
                "https://aistudio.google.com/apikey або запустіть із --provider fake."
            )
        self.model = model
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(timeout_sec * 1000)),
        )

    def _config(self, system: str, schema: dict[str, Any]) -> types.GenerateContentConfig:
        cfg: dict[str, Any] = {
            "system_instruction": system,
            # Класифікація — не творча задача. 0 прибирає всю варіативність,
            # яку взагалі можна прибрати з боку клієнта (але не всю взагалі —
            # див. розділ про недетермінізм у README).
            "temperature": 0.0,
            "response_mime_type": "application/json",
            "response_schema": schema,
            "candidate_count": 1,
        }
        # Вимикаємо «мислення»: воно оплачується як вихідні токени, а для
        # класифікації за готовою таксономією нічого не додає — роль ланцюжка
        # міркувань виконує поле reasoning, яке генерується першим.
        #
        # Тільки для 2.5: gemini-3.5-flash-lite відповідає на thinking_budget=0
        # помилкою 400, а в gemini-3.5-flash мислення увімкнене й вимкнути його
        # не можна — саме тому той самий файл коштує там у 15 разів більше
        # (порівняння в README).
        if self.model.startswith("gemini-2.5"):
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        return types.GenerateContentConfig(**cfg)

    async def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        schema: dict[str, Any],
    ) -> LLMResponse:
        contents = [
            types.Content(role=role, parts=[types.Part(text=text)]) for role, text in messages
        ]
        try:
            response = await self._client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=self._config(system, schema),
            )
        except genai_errors.APIError as exc:
            code = getattr(exc, "code", None)
            if code in _RETRYABLE_CODES:
                raise TransientLLMError(
                    f"Gemini {code}: {_short(exc)}", retry_after=_retry_after(exc)
                ) from exc
            raise FatalLLMError(f"Gemini {code}: {_short(exc)}") from exc
        except (TimeoutError, httpx.TransportError) as exc:
            # Обриви й таймаути нижче рівня SDK — теж випадковість, повторюємо.
            raise TransientLLMError(f"Мережевий збій при зверненні до Gemini: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            # Усе інше — не мережа, а наша або чужа помилка в коді чи конфігу.
            # Повторювати її безглуздо, тому Fatal: рядок стане `failed`,
            # прогін не впаде, причина буде видна у звіті.
            raise FatalLLMError(f"Неочікувана помилка клієнта Gemini: {exc!r}") from exc

        usage = response.usage_metadata
        text = response.text or ""
        if not text.strip():
            # Порожня відповідь = спрацював фільтр безпеки або обрізало ліміт.
            # Це не транспортна проблема: повтор дасть те саме.
            reason = getattr(response, "prompt_feedback", None)
            raise FatalLLMError(f"Gemini повернув порожню відповідь (feedback={reason})")

        return LLMResponse(
            text=text,
            prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=(getattr(usage, "candidates_token_count", 0) or 0)
            + (getattr(usage, "thoughts_token_count", 0) or 0),
        )

    async def aclose(self) -> None:
        # SDK тримає з'єднання сам; метод є заради єдиного контракту клієнтів.
        return None
