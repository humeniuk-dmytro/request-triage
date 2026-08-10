"""Клієнт-заглушка. Два режими, обидва детерміновані й без мережі.

**Скриптований** (`FakeClient(responses=[...])`) — віддає наперед задані сирі
відповіді, зокрема навмисно зіпсовані. Це те, чим тестується драбина валідації:
перевірити поведінку на битому JSON, дочекавшись його від справжньої моделі,
неможливо.

**Евристичний** (`FakeClient()`) — класифікує за ключовими словами. Він потрібен
не для якості, а щоб `--provider fake` давав рев'юеру повний робочий прогін без
жодного ключа. Це чесний baseline: наскільки LLM кращий за пошук підрядка,
видно, склавши два вивода поруч.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from .base import LLMResponse, Message

_KNOWN_SYSTEMS = [
    "Google Ads",
    "PlanFix",
    "Планфікс",
    "BigQuery",
    "Slack",
    "Telegram",
    "Google Doc",
    "Google Sheet",
    "Meta",
    "CRM",
    "1С",
    "Jira",
    "Notion",
]

_DEPARTMENTS = {
    "продаж": "продажі",
    "hr": "HR",
    "резюме": "HR",
    "бухгалтер": "бухгалтерія",
    "аналітик": "аналітика",
    "маркетинг": "маркетинг",
    "контент": "контент",
    "smm": "SMM",
    "підтримк": "підтримка",
}

# Порядок важливий: перше збіжне правило виграє.
_CATEGORY_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("поза скоупом", ("дякую", "дякуємо", "все працює", "ноутбук", "закупити", "заявку")),
    ("баг/підтримка", ("зламал", "не працює", "не парситься", "полагодити", "баг", "висять")),
    ("питання/консультація", ("чи можна взагалі", "теоретичн", "цікаво ваша думка", "це баг чи")),
    ("інтеграція", ("інтеграц", "автоматом створюва", "падало в", "з'єднати")),
    (
        "звіт/аналітика",
        ("звіт", "вивантаж", "дашборд", "метрик", "аналітик", "дайджест", "саммарі", "summary"),
    ),
    (
        "автоматизація",
        ("автоматизувати", "автоматизац", "руками", "рутин", "прискорити", "інструмент"),
    ),
]

_HIGH = ("горить", "терміново", "до вечора", "сьогодні", "зламал", "стоїть", "висять")
_LOW = ("не горить", "теоретичн", "цікаво", "ідея", "дякую", "поки що нічого")


def _extract_request_text(messages: Sequence[Message]) -> str:
    for _role, text in reversed(messages):
        match = re.search(r"<request_text>\n(.*?)\n</request_text>", text, re.DOTALL)
        if match:
            return match.group(1)
    return messages[-1][1] if messages else ""


def _heuristic_extraction(text: str) -> dict[str, Any]:
    low = text.lower()

    category = "питання/консультація"
    for name, markers in _CATEGORY_RULES:
        if any(m in low for m in markers):
            category = name
            break

    if any(m in low for m in _LOW):
        priority = "low"
    elif any(m in low for m in _HIGH):
        priority = "high"
    else:
        priority = "medium"

    department = next((v for k, v in _DEPARTMENTS.items() if k in low), None)
    systems = [s for s in _KNOWN_SYSTEMS if s.lower() in low]

    # Короткий текст без жодної конкретики — єдиний випадок, коли евристика
    # має право сказати «не зрозуміло».
    vague = len(text.split()) < 8 and not systems
    questions = (
        [
            "Який конкретно результат ви очікуєте?",
            "З якими даними або системами це має працювати?",
            "Наскільки це терміново і чи є дедлайн?",
        ]
        if vague
        else []
    )

    summary = " ".join(text.split())
    if len(summary) > 200:
        summary = summary[:197] + "…"

    return {
        "reasoning": f"Евристика: категорію обрано за ключовими словами, пріоритет — {priority}.",
        "category": category,
        "priority": priority,
        "target_department": department,
        "short_summary": summary or "порожній запит",
        "requested_actions": [] if vague or category == "поза скоупом" else [summary[:120]],
        "needs_clarification": vague,
        "clarification_questions": questions,
        "confidence": "low",  # евристика ніколи не має права бути впевненою
        "mentioned_systems": systems,
    }


class FakeClient:
    name = "fake"

    def __init__(
        self,
        responses: Sequence[str | Exception] | None = None,
        model: str = "fake-heuristic",
    ) -> None:
        self.model = model
        self._responses: list[str | Exception] | None = (
            list(responses) if responses is not None else None
        )
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        schema: dict[str, Any],
    ) -> LLMResponse:
        self.calls.append(list(messages))

        if self._responses is not None:
            if not self._responses:
                raise AssertionError("FakeClient: скрипт відповідей вичерпано")
            nxt = self._responses.pop(0)
            if isinstance(nxt, Exception):
                raise nxt
            return LLMResponse(text=nxt, prompt_tokens=100, output_tokens=50)

        if "duplicates" in schema.get("properties", {}):
            # Евристика не шукає дублів: це вимагає розуміння суті, а не збігу слів.
            payload: dict[str, Any] = {"duplicates": []}
        else:
            payload = _heuristic_extraction(_extract_request_text(messages))

        return LLMResponse(
            text=json.dumps(payload, ensure_ascii=False),
            prompt_tokens=0,
            output_tokens=0,
        )

    async def aclose(self) -> None:
        return None
