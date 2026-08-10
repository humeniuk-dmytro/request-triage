"""Спільні фікстури.

Жоден тест не ходить у мережу і не потребує ключа: усе через FakeClient.
Це не компроміс, а вимога — тести, які залежать від живої моделі, недетерміновані,
повільні, платні й не запускаються в CI.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from triage.models import InboxRow

_REAL_SLEEP = asyncio.sleep

VALID_PAYLOAD: dict[str, Any] = {
    "reasoning": "Просять прибрати ручну роботу, дедлайну немає.",
    "category": "автоматизація",
    "priority": "medium",
    "target_department": "маркетинг",
    "short_summary": "Автоматизувати щотижневий звіт по Google Ads.",
    "requested_actions": ["автоматизувати вивантаження звіту по Google Ads"],
    "needs_clarification": False,
    "clarification_questions": [],
    "confidence": "high",
    "mentioned_systems": ["Google Ads"],
}


def payload(**overrides: Any) -> str:
    """Валідна відповідь моделі з точковими змінами."""
    return json.dumps({**VALID_PAYLOAD, **overrides}, ensure_ascii=False)


@pytest.fixture
def row() -> InboxRow:
    return InboxRow(
        id="REQ-001",
        channel="Slack",
        timestamp="2026-06-08 09:14",
        raw_text="Можна автоматизувати щотижневий звіт по Google Ads?",
    )


@pytest.fixture(autouse=True)
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Витримки між повторами реальні в проді й безглузді в тестах.

    Замість очікування збираємо тривалості: так тести і швидкі, і можуть
    перевірити, що пауза була саме такою, як просив сервер.
    """
    recorded: list[float] = []

    async def instant(seconds: float) -> None:
        recorded.append(seconds)

    monkeypatch.setattr("triage.extractor.asyncio.sleep", instant)
    return recorded


@pytest.fixture
def real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Повертає справжній asyncio.sleep.

    Патч вище чіпляється до модуля asyncio, тобто діє глобально. Тестам, які
    міряють реальний темп обмежувача, потрібне справжнє очікування — вони
    просять цю фікстуру, і вона перекриває патч (autouse застосовується раніше).
    """
    monkeypatch.setattr(asyncio, "sleep", _REAL_SLEEP)
