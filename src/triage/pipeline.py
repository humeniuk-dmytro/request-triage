"""Читання інбоксу і паралельна обробка.

Обробка запитів — це I/O-очікування, а не обчислення, тому asyncio: 18 запитів
послідовно — це 18 повних round-trip'ів, паралельно — стільки, скільки дозволяє
rate limit. Паралельність обмежена семафором навмисно: безкоштовний тір Gemini
має жорсткий ліміт запитів на хвилину, і «швидше» тут дуже швидко стає «429».
"""

from __future__ import annotations

import asyncio
import csv
import logging
from collections.abc import Callable, Iterable
from pathlib import Path

from pydantic import ValidationError

from .extractor import extract_one_safe
from .llm.base import LLMClient
from .models import FailureKind, InboxRow, ProcessingMeta, Status, TriagedRequest

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"id", "channel", "timestamp", "raw_text"}


def read_inbox(path: Path) -> tuple[list[InboxRow], list[str]]:
    """CSV → рядки інбоксу + список проблем.

    Битий рядок не зупиняє читання файлу: він потрапляє у список проблем,
    який CLI показує користувачеві. Вхідні дані «як у реальному житті» —
    значить, у них бувають порожні поля й зайві колонки.
    """
    if not path.exists():
        raise FileNotFoundError(f"Вхідний файл не знайдено: {path}")

    rows: list[InboxRow] = []
    problems: list[str] = []
    seen_ids: set[str] = set()

    # utf-8-sig: вивантаження з таблиць регулярно приїжджають із BOM.
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"У {path.name} бракує обов'язкових колонок: {', '.join(sorted(missing))}"
            )

        for lineno, raw in enumerate(reader, start=2):
            try:
                row = InboxRow.model_validate(
                    {k: (v or "") for k, v in raw.items() if k in REQUIRED_COLUMNS}
                )
            except ValidationError as exc:
                problems.append(f"рядок {lineno}: {exc.errors()[0]['msg']}")
                continue
            if row.id in seen_ids:
                problems.append(f"рядок {lineno}: дубльований id {row.id}, рядок пропущено")
                continue
            seen_ids.add(row.id)
            rows.append(row)

    return rows, problems


class _CircuitBreaker:
    """Запобіжник на випадок зламаного конфіга.

    Невірний ключ або відсутній доступ до моделі провалить кожен запит.
    Без запобіжника ми акуратно, з витримками й повторами, дійдемо до кінця
    файлу — витративши квоту й час на гарантовано порожній результат.

    Рахує ЛИШЕ фатальні збої. Вичерпаний ліміт запитів — не зламаний конфіг:
    це нормальний стан безкоштовного тіру, який лікується темпом, а не
    зупинкою прогону. Перша версія цього не розрізняла й глушила прогін на
    першій же серії 429.
    """

    def __init__(self, threshold: int = 3) -> None:
        self._threshold = threshold
        self._consecutive = 0
        self.tripped = False

    def record(self, ok: bool) -> None:
        if ok:
            self._consecutive = 0
            return
        self._consecutive += 1
        if self._consecutive >= self._threshold and not self.tripped:
            self.tripped = True
            log.error(
                "%d запитів поспіль завершились помилкою — зупиняю прогін. "
                "Найімовірніша причина: ключ, доступ до моделі або мережа.",
                self._threshold,
            )


async def run_triage(
    client: LLMClient,
    rows: Iterable[InboxRow],
    *,
    concurrency: int = 5,
    max_retries: int = 3,
    max_repairs: int = 1,
    breaker_threshold: int = 3,
    on_done: Callable[[TriagedRequest], None] | None = None,
) -> list[TriagedRequest]:
    """Обробити всі рядки. Порядок результатів = порядок входу."""
    rows = list(rows)
    semaphore = asyncio.Semaphore(concurrency)
    breaker = _CircuitBreaker(breaker_threshold)

    async def worker(row: InboxRow) -> TriagedRequest:
        async with semaphore:
            if breaker.tripped:
                return TriagedRequest(
                    **row.model_dump(),
                    meta=ProcessingMeta(
                        status=Status.FAILED,
                        failure_kind=FailureKind.FATAL,
                        model=client.model,
                        error="прогін зупинено запобіжником після серії збоїв поспіль",
                    ),
                )
            result = await extract_one_safe(
                client, row, max_retries=max_retries, max_repairs=max_repairs
            )
        breaker.record(result.meta.failure_kind is not FailureKind.FATAL)
        if on_done is not None:
            on_done(result)
        return result

    # gather зберігає порядок за індексом незалежно від порядку завершення,
    # тому вивід відтворюваний і його можна порівнювати між прогонами.
    return list(await asyncio.gather(*(worker(row) for row in rows)))
