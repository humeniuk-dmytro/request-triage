"""Спільні фікстури.

Жоден тест не ходить у мережу і не потребує ключа: усе через FakeClient.
Це не компроміс, а вимога — тести, які залежать від живої моделі, недетерміновані,
повільні, платні й не запускаються в CI.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from triage.models import InboxRow

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
def no_backoff_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Витримки між повторами реальні в проді й безглузді в тестах."""

    async def instant(_seconds: float) -> None:
        return None

    monkeypatch.setattr("triage.extractor.asyncio.sleep", instant)
