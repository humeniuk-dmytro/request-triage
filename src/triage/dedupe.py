"""Другий прохід: пошук запитів, які просять те саме.

Навіщо окремий прохід. Класифікація йде порядково — модель бачить рівно один
запит і фізично не може знати, що три години тому хтось просив те саме. Тобто
дублі не «погано визначаються» в основному проході, а не визначаються взагалі.

Навіщо взагалі. В інбоксі 18 запитів REQ-013 дослівно повторює REQ-001 («той
самий звіт по Google Ads, що Оля просила в понеділок»). Саме на цьому реальні
черги втрачають час: дві людини роблять одну роботу.

Ціна — один виклик на весь прогін: на вхід ідуть тільки id та короткі саммарі,
не оригінальні тексти.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict, ValidationError

from .extractor import call_with_retry, parse_json_object
from .llm.base import FatalLLMError, LLMClient, TransientLLMError
from .models import Status, TriagedRequest
from .prompt import DEDUPE_RESPONSE_SCHEMA, DEDUPE_SYSTEM_PROMPT

log = logging.getLogger(__name__)


class DuplicatePair(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    reason: str = ""
    duplicate_id: str
    original_id: str


class _DedupeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    duplicates: list[DuplicatePair] = []


def _build_listing(results: list[TriagedRequest]) -> str:
    lines = []
    for r in results:
        if r.extraction is None:
            continue
        ts = r.timestamp.strftime("%Y-%m-%d %H:%M") if r.timestamp else "—"
        lines.append(f"{r.id} [{ts}] {r.extraction.short_summary}")
    return "\n".join(lines)


async def find_duplicates(
    client: LLMClient,
    results: list[TriagedRequest],
    *,
    max_retries: int = 3,
) -> tuple[list[DuplicatePair], int, int]:
    """Проставляє possible_duplicate_of у results. Повертає пари й витрачені токени.

    Ніколи не кидає винятків: дублі — це приємний бонус до звіту, і його
    відсутність не привід втрачати вже отриману класифікацію.
    """
    listing = _build_listing(results)
    if listing.count("\n") < 1:  # менше двох запитів — нема чого порівнювати
        return [], 0, 0

    try:
        response, _ = await call_with_retry(
            client,
            system=DEDUPE_SYSTEM_PROMPT,
            messages=[("user", listing)],
            schema=DEDUPE_RESPONSE_SCHEMA,
            max_retries=max_retries,
        )
    except (TransientLLMError, FatalLLMError) as exc:
        log.warning("прохід по дублях не вдався: %s", exc)
        return [], 0, 0

    try:
        parsed = _DedupeResponse.model_validate(parse_json_object(response.text))
    except (ValueError, ValidationError) as exc:
        log.warning("невалідна відповідь проходу по дублях: %s", exc)
        return [], response.prompt_tokens, response.output_tokens

    by_id = {r.id: r for r in results}
    accepted: list[DuplicatePair] = []

    for pair in parsed.duplicates:
        dup, orig = by_id.get(pair.duplicate_id), by_id.get(pair.original_id)
        # Модель охоче вигадує id, які їй ніхто не показував, і зрідка
        # оголошує запит дублем самого себе. Перевіряємо, а не довіряємо.
        if dup is None or orig is None or dup.id == orig.id:
            log.warning(
                "відкидаю пару дублів %s → %s: id не існує або збігаються",
                pair.duplicate_id,
                pair.original_id,
            )
            continue
        if dup.meta.status is Status.FAILED or orig.meta.status is Status.FAILED:
            continue
        # Дублем вважаємо пізніший запит. Якщо модель переплутала напрям —
        # розвертаємо, а не відкидаємо: сама пара знайдена правильно.
        if dup.timestamp and orig.timestamp and dup.timestamp < orig.timestamp:
            dup, orig = orig, dup
            pair = DuplicatePair(reason=pair.reason, duplicate_id=dup.id, original_id=orig.id)
        dup.possible_duplicate_of = orig.id
        accepted.append(pair)

    return accepted, response.prompt_tokens, response.output_tokens
