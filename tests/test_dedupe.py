"""Прохід по дублях: перевіряємо не лише те, що пари знаходяться,
а насамперед те, що вигадані моделлю пари не потрапляють у звіт.
"""

from __future__ import annotations

import json

from triage.dedupe import find_duplicates
from triage.llm.fake import FakeClient
from triage.models import Extraction, ProcessingMeta, Status, TriagedRequest

from .conftest import VALID_PAYLOAD


def _req(rid: str, summary: str, ts: str) -> TriagedRequest:
    return TriagedRequest(
        id=rid,
        channel="Slack",
        timestamp=ts,
        raw_text=summary,
        extraction=Extraction.model_validate({**VALID_PAYLOAD, "short_summary": summary}),
        meta=ProcessingMeta(status=Status.OK, model="test"),
    )


def _dedupe_response(*pairs: tuple[str, str]) -> str:
    return json.dumps(
        {
            "duplicates": [
                {"duplicate_id": d, "original_id": o, "reason": "та сама робота"} for d, o in pairs
            ]
        },
        ensure_ascii=False,
    )


async def test_duplicate_is_linked_to_the_earlier_request() -> None:
    results = [
        _req("REQ-001", "Автоматизувати щотижневий звіт по Google Ads", "2026-06-08 09:14"),
        _req("REQ-013", "Той самий звіт по Google Ads ще одному колезі", "2026-06-08 16:02"),
    ]
    client = FakeClient(responses=[_dedupe_response(("REQ-013", "REQ-001"))])

    pairs, _in, _out = await find_duplicates(client, results)

    assert [(p.duplicate_id, p.original_id) for p in pairs] == [("REQ-013", "REQ-001")]
    assert results[1].possible_duplicate_of == "REQ-001"
    assert results[0].possible_duplicate_of is None


async def test_reversed_direction_is_flipped_not_dropped() -> None:
    """Пара знайдена правильно, переплутано лише напрям — розвертаємо."""
    results = [
        _req("REQ-001", "Звіт по Google Ads", "2026-06-08 09:14"),
        _req("REQ-013", "Той самий звіт по Google Ads", "2026-06-08 16:02"),
    ]
    client = FakeClient(responses=[_dedupe_response(("REQ-001", "REQ-013"))])

    pairs, _in, _out = await find_duplicates(client, results)

    assert (pairs[0].duplicate_id, pairs[0].original_id) == ("REQ-013", "REQ-001")
    assert results[1].possible_duplicate_of == "REQ-001"


async def test_hallucinated_ids_are_discarded() -> None:
    """Модель охоче називає id, яких їй ніхто не показував."""
    results = [
        _req("REQ-001", "Звіт по Google Ads", "2026-06-08 09:14"),
        _req("REQ-002", "Інтеграція Slack з PlanFix", "2026-06-08 10:00"),
    ]
    client = FakeClient(
        responses=[_dedupe_response(("REQ-777", "REQ-001"), ("REQ-002", "REQ-002"))]
    )

    pairs, _in, _out = await find_duplicates(client, results)

    assert pairs == []
    assert all(r.possible_duplicate_of is None for r in results)


async def test_invalid_response_does_not_break_the_run() -> None:
    """Дублі — бонус до звіту; їх втрата не має коштувати класифікації."""
    results = [
        _req("REQ-001", "Звіт по Google Ads", "2026-06-08 09:14"),
        _req("REQ-002", "Інтеграція Slack з PlanFix", "2026-06-08 10:00"),
    ]
    client = FakeClient(responses=["{ не json"])

    pairs, _in, _out = await find_duplicates(client, results)
    assert pairs == []


async def test_single_request_costs_nothing() -> None:
    """Порівнювати нема з чим — виклику не має бути взагалі."""
    client = FakeClient(responses=[])
    pairs, _in, _out = await find_duplicates(client, [_req("REQ-001", "Звіт", "2026-06-08 09:14")])
    assert pairs == []
    assert client.calls == []
