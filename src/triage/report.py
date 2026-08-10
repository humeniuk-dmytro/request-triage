"""Агрегати й вихідні файли.

Агрегати рахує код, а не модель. Запитувати в LLM «скільки тут запитів із
пріоритетом high» — це платити за арифметику й отримувати ймовірнісну
відповідь там, де можлива точна.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from .dedupe import DuplicatePair
from .models import (
    Aggregates,
    Category,
    Confidence,
    Priority,
    RunResult,
    Status,
    TriagedRequest,
)

NOT_SET = "не визначено"


def compute_aggregates(
    results: list[TriagedRequest],
    duplicates: list[DuplicatePair] | None = None,
) -> Aggregates:
    duplicates = duplicates or []
    done = [(r, r.extraction) for r in results if r.extraction is not None]
    extractions = [e for _r, e in done]

    by_category = Counter(e.category.value for e in extractions)
    by_priority = Counter(e.priority.value for e in extractions)
    by_confidence = Counter(e.confidence.value for e in extractions)
    by_department = Counter(e.target_department or NOT_SET for e in extractions)
    by_channel = Counter(r.channel or NOT_SET for r in results)
    by_status = Counter(r.meta.status.value for r in results)

    def ordered(counter: Counter, order: list[str]) -> dict[str, int]:
        """Фіксований порядок довідників: звіти між прогонами лишаються порівнюваними."""
        return {key: counter.get(key, 0) for key in order}

    def by_size(counter: Counter) -> dict[str, int]:
        return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))

    return Aggregates(
        total=len(results),
        by_category=ordered(by_category, [c.value for c in Category]),
        by_priority=ordered(by_priority, [p.value for p in Priority]),
        by_department=by_size(by_department),
        by_channel=by_size(by_channel),
        by_status=ordered(by_status, [s.value for s in Status]),
        by_confidence=ordered(by_confidence, [c.value for c in Confidence]),
        needs_clarification_ids=[r.id for r, e in done if e.needs_clarification],
        duplicate_pairs=[(d.duplicate_id, d.original_id) for d in duplicates],
        failed_ids=[r.id for r in results if r.meta.status is Status.FAILED],
    )


def _table(title: str, counts: dict[str, int], total: int) -> list[str]:
    lines = [f"### {title}", "", "| Значення | Кількість | Частка |", "|---|---:|---:|"]
    for key, value in counts.items():
        share = f"{value / total * 100:.0f}%" if total else "—"
        lines += [f"| {key} | {value} | {share} |"]
    return [*lines, ""]


def render_report_md(result: RunResult, duplicates: list[DuplicatePair]) -> str:
    m, agg = result.manifest, result.aggregates
    by_id = {r.id: r for r in result.requests}

    out: list[str] = [
        "# Звіт по вхідних запитах",
        "",
        f"Файл: `{m.input_file}` · запитів: **{m.requests_total}** · "
        f"модель: `{m.model}` (провайдер `{m.provider}`)",
        f"Прогін: {m.started_at:%Y-%m-%d %H:%M:%S} · {m.wall_time_sec:.1f} с · "
        f"{m.total_tokens} токенів · оцінка вартості ${m.estimated_cost_usd:.4f}",
        "",
    ]

    out += _table("За категоріями", agg.by_category, m.requests_total)
    out += _table("За пріоритетом", agg.by_priority, m.requests_total)
    out += _table("За відділами", agg.by_department, m.requests_total)
    out += _table("За каналами", agg.by_channel, m.requests_total)
    out += _table("Впевненість моделі", agg.by_confidence, m.requests_total)
    out += _table("Статус обробки", agg.by_status, m.requests_total)

    # Найтерміновіше — окремо й першим по суті: саме це беруть у роботу сьогодні.
    urgent = [r for r in result.requests if r.extraction and r.extraction.priority is Priority.HIGH]
    out += ["### Потребують реакції сьогодні (priority: high)", ""]
    if urgent:
        out += ["| ID | Канал | Категорія | Суть |", "|---|---|---|---|"]
        out += [
            f"| {r.id} | {r.channel} | {r.extraction.category.value} | "
            f"{r.extraction.short_summary} |"
            for r in urgent
            if r.extraction
        ]
    else:
        out += ["Немає."]
    out += [""]

    out += ["### Потребують уточнення", ""]
    if agg.needs_clarification_ids:
        for rid in agg.needs_clarification_ids:
            r = by_id[rid]
            assert r.extraction is not None
            out += [f"**{rid}** ({r.channel}) — {r.extraction.short_summary}", ""]
            out += [f"- {q}" for q in r.extraction.clarification_questions]
            out += [""]
    else:
        out += ["Немає.", ""]

    out += ["### Можливі дублі", ""]
    if duplicates:
        out += ["| Дубль | Оригінал | Чому |", "|---|---|---|"]
        out += [f"| {d.duplicate_id} | {d.original_id} | {d.reason} |" for d in duplicates]
    else:
        out += ["Не знайдено."]
    out += [""]

    out += ["### Збої обробки", ""]
    if agg.failed_ids:
        out += ["| ID | Помилка |", "|---|---|"]
        out += [f"| {rid} | {by_id[rid].meta.error or '—'} |" for rid in agg.failed_ids]
        out += [
            "",
            "Ці рядки не втрачені: вони є в `output.json` зі статусом `failed` "
            "і повним текстом помилки.",
        ]
    else:
        out += ["Немає — усі запити оброблено."]
    out += [""]

    return "\n".join(out)


CSV_COLUMNS = [
    "id",
    "channel",
    "timestamp",
    "status",
    "category",
    "priority",
    "target_department",
    "needs_clarification",
    "confidence",
    "short_summary",
    "requested_actions",
    "clarification_questions",
    "mentioned_systems",
    "possible_duplicate_of",
    "reasoning",
]


def to_rows(results: list[TriagedRequest]) -> list[dict[str, str]]:
    """Пласке табличне представлення — для CSV і для Google Sheets."""
    rows = []
    for r in results:
        e = r.extraction
        rows.append(
            {
                "id": r.id,
                "channel": r.channel,
                "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M") if r.timestamp else "",
                "status": r.meta.status.value,
                "category": e.category.value if e else "",
                "priority": e.priority.value if e else "",
                "target_department": (e.target_department or "") if e else "",
                "needs_clarification": ("так" if e.needs_clarification else "ні") if e else "",
                "confidence": e.confidence.value if e else "",
                "short_summary": e.short_summary if e else (r.meta.error or ""),
                "requested_actions": " | ".join(e.requested_actions) if e else "",
                "clarification_questions": " | ".join(e.clarification_questions) if e else "",
                "mentioned_systems": ", ".join(e.mentioned_systems) if e else "",
                "possible_duplicate_of": r.possible_duplicate_of or "",
                "reasoning": e.reasoning if e else "",
            }
        )
    return rows


def write_outputs(
    out_dir: Path,
    result: RunResult,
    duplicates: list[DuplicatePair],
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": out_dir / "output.json",
        "report": out_dir / "report.md",
        "csv": out_dir / "output.csv",
    }

    paths["json"].write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["report"].write_text(render_report_md(result, duplicates), encoding="utf-8")

    with paths["csv"].open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(to_rows(result.requests))

    return paths
