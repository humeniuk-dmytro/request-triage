"""Обробка одного запиту: від тексту до провалідованої структури.

Тут живе вся стійкість до поведінки моделі. Драбина знизу вгору:

1. Схема на рівні API — модель обмежена ще до генерації (llm/gemini.py).
2. Толерантний розбір — зрізаємо ```-огорожі й прозу навколо JSON.
3. Нормалізація до валідації — «Автоматизація», «HIGH », «так», рядок замість
   списку приводяться до контракту без звернення до моделі (models.py).
4. Строга валідація Pydantic, разом із семантичними правилами.
5. Один repair-виклик із дослівним текстом помилки.
6. Запис зі статусом `failed` — рядок зберігається, прогін не падає.

Окремо і принципово інакше обробляються транспортні збої: 429/5xx/таймаут —
це випадковість, яка лікується повтором того самого запиту з витримкою.
Невалідний вивід повтором того самого запиту не лікується ніколи.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from typing import Any

from pydantic import ValidationError

from .llm.base import (
    FatalLLMError,
    LLMClient,
    LLMResponse,
    Message,
    TransientLLMError,
)
from .models import (
    GEMINI_RESPONSE_SCHEMA,
    Extraction,
    FailureKind,
    InboxRow,
    ProcessingMeta,
    Status,
    TriagedRequest,
)
from .prompt import SYSTEM_PROMPT, build_repair_message, build_user_message

log = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_json_object(text: str) -> dict[str, Any]:
    """Розбір відповіді моделі в об'єкт.

    Навіть із response_mime_type=application/json трапляється огорожа ```json
    або пояснення навколо. Відновлюємо те, що відновлюється однозначно;
    все інше — чесна помилка, яка йде в repair.
    """
    cleaned = _FENCE_RE.sub("", text.strip())
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Остання спроба: вирізати найширший блок від першої { до останньої }.
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"Відповідь не містить JSON-об'єкта: {text[:200]!r}") from None
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Невалідний JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"Очікувався JSON-об'єкт, отримано {type(parsed).__name__}")
    return parsed


def _format_validation_error(exc: ValidationError) -> str:
    """Помилки Pydantic у вигляді, зрозумілому моделі: поле → що не так."""
    lines = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "(корінь)"
        lines.append(f"- {loc}: {err['msg']}")
    return "\n".join(lines)


async def call_with_retry(
    client: LLMClient,
    *,
    system: str,
    messages: list[Message],
    schema: dict[str, Any],
    max_retries: int,
) -> tuple[LLMResponse, int]:
    """Виклик моделі з витримкою на транспортних збоях.

    Експонента з джитером: без джитера всі паралельні задачі, що впіймали 429,
    прокидаються одночасно й отримують 429 ще раз.

    Повертає відповідь і кількість фактичних звернень до API.
    """
    attempts = 0
    delay = 1.0
    while True:
        attempts += 1
        try:
            return await client.complete(system=system, messages=messages, schema=schema), attempts
        except TransientLLMError as exc:
            if attempts > max_retries:
                raise
            # Якщо сервер сам назвав витримку — слухаємо його. Наша експонента
            # проти «retry in 28s» дає лише ще одну відмову і ще один запит
            # у ліміт, який щойно вичерпано.
            sleep_for = (
                exc.retry_after + random.uniform(0, 1)
                if exc.retry_after is not None
                else delay + random.uniform(0, delay / 2)
            )
            log.warning(
                "тимчасовий збій (%s), спроба %d/%d, пауза %.1fс",
                exc,
                attempts,
                max_retries + 1,
                sleep_for,
            )
            await asyncio.sleep(sleep_for)
            delay *= 2


async def extract_one(
    client: LLMClient,
    row: InboxRow,
    *,
    max_retries: int = 3,
    max_repairs: int = 1,
) -> TriagedRequest:
    """Один рядок інбоксу → один запис результату.

    Не кидає винятків, крім FatalLLMError: будь-яка проблема з конкретним
    запитом стає записом зі статусом `failed`, а не втраченим рядком.
    """
    started = time.perf_counter()
    meta = ProcessingMeta(status=Status.FAILED, model=client.model)

    if not row.raw_text.strip():
        # Порожній текст класифікувати нема з чого — і платити за це теж нема за що.
        meta.status = Status.SKIPPED
        meta.error = "порожній raw_text — звернення до моделі не виконувалось"
        return TriagedRequest(**row.model_dump(), meta=meta)

    messages: list[Message] = [("user", build_user_message(row))]
    last_error = "невідома помилка"

    for repair_no in range(max_repairs + 1):
        try:
            response, attempts = await call_with_retry(
                client,
                system=SYSTEM_PROMPT,
                messages=messages,
                schema=GEMINI_RESPONSE_SCHEMA,
                max_retries=max_retries,
            )
        except TransientLLMError as exc:
            meta.failure_kind = FailureKind.TRANSIENT
            meta.error = f"вичерпано повтори після транспортних збоїв: {exc}"
            break

        meta.attempts += attempts
        meta.prompt_tokens += response.prompt_tokens
        meta.output_tokens += response.output_tokens

        try:
            extraction = Extraction.model_validate(parse_json_object(response.text))
        except (ValueError, ValidationError) as exc:
            last_error = (
                _format_validation_error(exc) if isinstance(exc, ValidationError) else str(exc)
            )
            log.info("невалідний вивід для %s (спроба %d): %s", row.id, repair_no + 1, last_error)
            if repair_no == max_repairs:
                meta.failure_kind = FailureKind.VALIDATION
                meta.error = f"вивід не пройшов валідацію: {last_error}"
                break
            # Показуємо моделі її ж відповідь і конкретну претензію до неї.
            messages += [("model", response.text), ("user", build_repair_message(last_error))]
            meta.repairs += 1
            continue

        meta.status = Status.REPAIRED if repair_no else Status.OK
        meta.latency_ms = int((time.perf_counter() - started) * 1000)
        return TriagedRequest(**row.model_dump(), extraction=extraction, meta=meta)

    meta.status = Status.FAILED
    meta.latency_ms = int((time.perf_counter() - started) * 1000)
    return TriagedRequest(**row.model_dump(), meta=meta)


async def extract_one_safe(
    client: LLMClient,
    row: InboxRow,
    *,
    max_retries: int = 3,
    max_repairs: int = 1,
) -> TriagedRequest:
    """Те саме, але FatalLLMError теж стає записом `failed`.

    Використовується конвеєром: один зламаний запит не має права зупинити
    обробку решти. Рішення «зупинити весь прогін» ухвалюється рівнем вище,
    на підставі кількості поспіль фатальних збоїв.
    """
    try:
        return await extract_one(client, row, max_retries=max_retries, max_repairs=max_repairs)
    except FatalLLMError as exc:
        return TriagedRequest(
            **row.model_dump(),
            meta=ProcessingMeta(
                status=Status.FAILED,
                failure_kind=FailureKind.FATAL,
                model=client.model,
                attempts=1,
                error=f"фатальна помилка звернення до моделі: {exc}",
            ),
        )
