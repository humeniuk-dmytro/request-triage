"""Дайджест підсумків у Telegram.

Навмисно на httpx, а не на aiogram: тут немає ані бота, який слухає апдейти,
ані станів, ані хендлерів — є один POST. Тягнути заради нього фреймворк —
це залежність, яку доведеться оновлювати роками заради одного рядка.

Не має права завалити прогін: результат уже записаний на диск, а розсилка —
це доставка. Немає токена — крок пропускається з попередженням.
"""

from __future__ import annotations

import logging

import httpx

from ..config import Settings
from ..dedupe import DuplicatePair
from ..models import RunResult

log = logging.getLogger(__name__)

API_URL = "https://api.telegram.org/bot{token}/sendMessage"
TELEGRAM_LIMIT = 4096


def render_digest(result: RunResult, duplicates: list[DuplicatePair]) -> str:
    agg, m = result.aggregates, result.manifest
    by_id = {r.id: r for r in result.requests}

    lines = [
        "<b>Тріаж вхідних запитів</b>",
        f"Оброблено: {agg.total} · {m.wall_time_sec:.1f} с · ~${m.estimated_cost_usd:.4f}",
        "",
        "<b>Категорії</b>",
    ]
    lines += [f"· {k}: {v}" for k, v in agg.by_category.items() if v]
    lines += ["", "<b>Пріоритет</b>"]
    lines += [f"· {k}: {v}" for k, v in agg.by_priority.items() if v]

    urgent = [r for r in result.requests if r.extraction and r.extraction.priority == "high"]
    if urgent:
        lines += ["", "<b>Сьогодні (high)</b>"]
        lines += [f"· <b>{r.id}</b> — {r.extraction.short_summary}" for r in urgent if r.extraction]

    if agg.needs_clarification_ids:
        lines += ["", "<b>Потребують уточнення</b>"]
        for rid in agg.needs_clarification_ids:
            e = by_id[rid].extraction
            if e:
                lines.append(f"· <b>{rid}</b> — {e.short_summary}")

    if duplicates:
        lines += ["", "<b>Можливі дублі</b>"]
        lines += [f"· {d.duplicate_id} ≈ {d.original_id}" for d in duplicates]

    if agg.failed_ids:
        lines += ["", f"<b>Збої обробки:</b> {', '.join(agg.failed_ids)}"]

    text = "\n".join(lines)
    if len(text) > TELEGRAM_LIMIT:
        text = text[: TELEGRAM_LIMIT - 20].rsplit("\n", 1)[0] + "\n…"
    return text


async def send_digest(
    settings: Settings,
    result: RunResult,
    duplicates: list[DuplicatePair],
) -> bool:
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        log.warning("--notify вказано, але TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID порожні — пропускаю")
        return False

    payload = {
        "chat_id": settings.telegram_chat_id,
        "text": render_digest(result, duplicates),
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        async with httpx.AsyncClient(timeout=15) as http:
            response = await http.post(
                API_URL.format(token=settings.telegram_bot_token), json=payload
            )
            response.raise_for_status()
    except httpx.HTTPError as exc:
        log.error("не вдалося надіслати дайджест у Telegram: %s", exc)
        return False

    log.info("дайджест надіслано в Telegram")
    return True
