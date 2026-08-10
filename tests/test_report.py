"""Агрегати й вихідні файли."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from triage.dedupe import DuplicatePair
from triage.models import (
    Extraction,
    ProcessingMeta,
    RunManifest,
    RunResult,
    Status,
    TriagedRequest,
)
from triage.report import compute_aggregates, render_report_md, write_outputs

from .conftest import VALID_PAYLOAD


def _req(
    rid: str, *, status: Status = Status.OK, channel: str = "Slack", **fields
) -> TriagedRequest:
    extraction = (
        Extraction.model_validate({**VALID_PAYLOAD, **fields})
        if status in (Status.OK, Status.REPAIRED)
        else None
    )
    return TriagedRequest(
        id=rid,
        channel=channel,
        timestamp="2026-06-08 09:14",
        raw_text="текст запиту",
        extraction=extraction,
        meta=ProcessingMeta(
            status=status,
            model="test",
            error="вивід не пройшов валідацію" if status is Status.FAILED else None,
        ),
    )


def _sample() -> list[TriagedRequest]:
    return [
        _req("REQ-001", category="автоматизація", priority="high", target_department="маркетинг"),
        _req(
            "REQ-002",
            category="автоматизація",
            priority="low",
            target_department=None,
            needs_clarification=True,
            clarification_questions=["Що саме автоматизувати?"],
        ),
        _req(
            "REQ-003",
            category="інтеграція",
            priority="high",
            target_department="продажі",
            channel="Email",
        ),
        _req("REQ-004", status=Status.FAILED, channel="Telegram"),
        _req("REQ-005", status=Status.SKIPPED, channel="Telegram"),
    ]


def test_aggregates_count_only_classified_requests() -> None:
    agg = compute_aggregates(_sample())

    assert agg.total == 5
    # 5 рядків, але класифіковано лише 3 — категорії не мають рахувати збої.
    assert sum(agg.by_category.values()) == 3
    assert agg.by_category["автоматизація"] == 2
    assert agg.by_category["інтеграція"] == 1
    assert agg.by_priority == {"low": 1, "medium": 0, "high": 2}
    assert agg.by_status == {"ok": 3, "repaired": 0, "failed": 1, "skipped": 1}
    assert agg.failed_ids == ["REQ-004"]


def test_all_categories_present_even_when_zero() -> None:
    """Порожня категорія — теж інформація; звіти між прогонами лишаються порівнюваними."""
    agg = compute_aggregates(_sample())
    assert list(agg.by_category) == [
        "автоматизація",
        "інтеграція",
        "звіт/аналітика",
        "баг/підтримка",
        "питання/консультація",
        "поза скоупом",
    ]


def test_missing_department_is_labelled_not_dropped() -> None:
    agg = compute_aggregates(_sample())
    assert agg.by_department["не визначено"] == 1


def test_clarification_list_matches_flags() -> None:
    agg = compute_aggregates(_sample())
    assert agg.needs_clarification_ids == ["REQ-002"]


def _run_result(results: list[TriagedRequest], duplicates: list[DuplicatePair]) -> RunResult:
    now = datetime(2026, 6, 8, 12, 0, 0)
    return RunResult(
        manifest=RunManifest(
            started_at=now,
            finished_at=now,
            provider="fake",
            model="test",
            prompt_sha256="deadbeef",
            input_file="data/input_requests.csv",
            requests_total=len(results),
            concurrency=5,
            prompt_tokens=100,
            output_tokens=50,
            total_tokens=150,
            estimated_cost_usd=0.0001,
            wall_time_sec=1.5,
        ),
        aggregates=compute_aggregates(results, duplicates),
        requests=results,
    )


def test_report_surfaces_what_a_human_acts_on() -> None:
    results = _sample()
    duplicates = [DuplicatePair(duplicate_id="REQ-003", original_id="REQ-001", reason="те саме")]
    md = render_report_md(_run_result(results, duplicates), duplicates)

    assert "Потребують реакції сьогодні" in md
    assert "REQ-002" in md and "Що саме автоматизувати?" in md  # питання, а не лише прапорець
    assert "REQ-003" in md and "REQ-001" in md  # пара дублів
    assert "вивід не пройшов валідацію" in md  # збій видно, а не сховано


def test_write_outputs_produces_all_three_files(tmp_path: Path) -> None:
    results = _sample()
    result = _run_result(results, [])
    paths = write_outputs(tmp_path, result, [])

    assert set(paths) == {"json", "report", "csv"}

    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert len(payload["requests"]) == 5
    assert payload["manifest"]["prompt_sha256"] == "deadbeef"
    # Збійний рядок є у виводі з поясненням — саме це вимагає ТЗ.
    failed = next(r for r in payload["requests"] if r["id"] == "REQ-004")
    assert failed["extraction"] is None
    assert failed["meta"]["status"] == "failed"
    assert failed["raw_text"] == "текст запиту"

    with paths["csv"].open(encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 5
    assert rows[0]["category"] == "автоматизація"
    assert rows[1]["needs_clarification"] == "так"
