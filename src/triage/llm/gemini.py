"""Клієнт Gemini (безкоштовний тір AI Studio).

Обмеження виводу тут дворівневе ще до нашої валідації:
`response_mime_type="application/json"` прибирає прозу навколо JSON, а
`response_schema` не дає моделі вигадати поля чи значення enum. Це не робить
валідацію зайвою — це робить її рідше потрібною, а отже дешевшою.
"""

from __future__ import annotations

from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from .base import FatalLLMError, LLMResponse, Message, TransientLLMError

# Коди, які лікуються повтором того самого запиту.
_RETRYABLE_CODES = {408, 409, 429, 500, 502, 503, 504}


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
        # Вимикаємо «мислення»: воно оплачується як вихідні токени й для
        # класифікації за готовою таксономією нічого не додає — роль ланцюжка
        # міркувань виконує поле reasoning, яке генерується першим.
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
                raise TransientLLMError(f"Gemini {code}: {exc}") from exc
            raise FatalLLMError(f"Gemini {code}: {exc}") from exc
        except TimeoutError as exc:
            raise TransientLLMError(f"Таймаут звернення до Gemini: {exc}") from exc
        except Exception as exc:  # обриви з'єднання нижче рівня SDK
            raise TransientLLMError(f"Мережевий збій при зверненні до Gemini: {exc}") from exc

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
