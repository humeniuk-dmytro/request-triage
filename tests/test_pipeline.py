"""Читання інбоксу і паралельна обробка."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from triage.llm.base import FatalLLMError, LLMResponse, Message
from triage.llm.fake import FakeClient
from triage.models import InboxRow, Status
from triage.pipeline import read_inbox, run_triage

from .conftest import VALID_PAYLOAD, payload

DATA = Path(__file__).resolve().parents[1] / "data" / "input_requests.csv"


class ByTextClient:
    """Відповідає залежно від тексту запиту, а не від порядку виклику.

    FakeClient зі списком відповідей віддає їх послідовно — під паралельною
    обробкою це означає, що відповідь дістанеться випадковому запиту. Тут
    прив'язка жорстка, тому тест перевіряє конвеєр, а не збіг таймінгів.
    """

    name = "by-text"
    model = "test"

    def __init__(self, rules: dict[str, str | Exception], delays: dict[str, float] | None = None):
        self._rules = rules
        self._delays = delays or {}

    async def complete(
        self, *, system: str, messages: list[Message], schema: dict[str, Any]
    ) -> LLMResponse:
        text = messages[0][1]
        for marker, response in self._rules.items():
            if marker in text:
                await asyncio.sleep(self._delays.get(marker, 0))
                if isinstance(response, Exception):
                    raise response
                return LLMResponse(text=response, prompt_tokens=10, output_tokens=5)
        raise AssertionError(f"немає правила для {text!r}")

    async def aclose(self) -> None:
        return None


# ------------------------------- читання CSV -------------------------------- #


def test_reads_real_inbox() -> None:
    rows, problems = read_inbox(DATA)
    assert len(rows) == 18
    assert problems == []
    assert rows[0].id == "REQ-001"
    assert rows[0].timestamp is not None and rows[0].timestamp.hour == 9


def test_missing_column_is_a_hard_error(tmp_path: Path) -> None:
    """Немає raw_text — класифікувати нема чого, це не «проблемний рядок»."""
    bad = tmp_path / "bad.csv"
    bad.write_text("id,channel\nREQ-1,Slack\n", encoding="utf-8")
    with pytest.raises(ValueError, match="raw_text"):
        read_inbox(bad)


def test_duplicate_id_is_reported_not_silently_kept(tmp_path: Path) -> None:
    src = tmp_path / "dup.csv"
    src.write_text(
        "id,channel,timestamp,raw_text\n"
        "REQ-1,Slack,2026-06-08 09:00,перший\n"
        "REQ-1,Slack,2026-06-08 09:01,другий\n",
        encoding="utf-8",
    )
    rows, problems = read_inbox(src)
    assert len(rows) == 1
    assert any("дубльований id" in p for p in problems)


def test_bom_and_empty_fields_survive(tmp_path: Path) -> None:
    """Вивантаження з таблиць приїжджають із BOM і порожніми клітинками."""
    src = tmp_path / "bom.csv"
    src.write_text("﻿id,channel,timestamp,raw_text\nREQ-1,,,\n", encoding="utf-8")
    rows, problems = read_inbox(src)
    assert problems == []
    assert rows[0].id == "REQ-1"
    assert rows[0].timestamp is None


def test_missing_file_is_explicit(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_inbox(tmp_path / "нема.csv")


# ---------------------------- паралельна обробка ---------------------------- #


def _rows(n: int) -> list[InboxRow]:
    return [
        InboxRow(id=f"REQ-{i:03d}", channel="Slack", raw_text=f"запит номер {i}")
        for i in range(1, n + 1)
    ]


async def test_order_is_preserved_regardless_of_completion_order() -> None:
    """Перший запит відповідає найповільніше — порядок виводу не має змінитись."""
    client = ByTextClient(
        rules={f"номер {i}": payload() for i in range(1, 4)},
        delays={"номер 1": 0.05, "номер 2": 0.01},
    )
    results = await run_triage(client, _rows(3), concurrency=3)
    assert [r.id for r in results] == ["REQ-001", "REQ-002", "REQ-003"]


async def test_one_bad_request_does_not_sink_the_run() -> None:
    client = ByTextClient(
        rules={
            "номер 1": payload(),
            "номер 2": "{ поламано",
            "номер 3": payload(),
        }
    )
    results = await run_triage(client, _rows(3), concurrency=3, max_repairs=0)

    statuses = {r.id: r.meta.status for r in results}
    assert statuses["REQ-001"] is Status.OK
    assert statuses["REQ-002"] is Status.FAILED
    assert statuses["REQ-003"] is Status.OK
    assert len(results) == 3  # жоден рядок не зник


async def test_concurrency_limit_is_respected() -> None:
    """Ліміт існує заради rate limit безкоштовного тіру, і він має працювати."""
    peak = 0
    active = 0

    class CountingClient(FakeClient):
        async def complete(self, **kwargs: Any) -> LLMResponse:
            nonlocal peak, active
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return LLMResponse(text=json.dumps(VALID_PAYLOAD, ensure_ascii=False))

    await run_triage(CountingClient(), _rows(10), concurrency=3)
    assert peak <= 3


async def test_breaker_stops_the_run_after_a_series_of_failures() -> None:
    """Зламаний ключ не має тихо з'їсти квоту на всьому файлі."""
    client = ByTextClient(
        rules={f"номер {i}": FatalLLMError("401 invalid key") for i in range(1, 9)}
    )
    results = await run_triage(client, _rows(8), concurrency=1, breaker_threshold=3)

    assert all(r.meta.status is Status.FAILED for r in results)
    stopped = [r for r in results if r.meta.error and "запобіжником" in r.meta.error]
    assert stopped, "після серії фатальних збоїв решта запитів мала припинитись"
    assert len(stopped) == 5  # перші три спробували, решта зупинена
