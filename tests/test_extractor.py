"""Драбина стійкості до поведінки моделі.

Кожен тест тут відповідає одному щаблю з extractor.py. Це головна частина
вимоги ТЗ «коректно обробляє випадки, коли модель повернула невалідні дані»:
таку поведінку неможливо перевірити на живій моделі — вона ламається тоді,
коли їй заманеться, а не тоді, коли ми тестуємо.
"""

from __future__ import annotations

from triage.extractor import extract_one, extract_one_safe, parse_json_object
from triage.llm.base import FatalLLMError, TransientLLMError
from triage.llm.fake import FakeClient
from triage.models import Category, InboxRow, Status

from .conftest import payload

# --------------------------- щабель 2: розбір ------------------------------ #


def test_parse_strips_code_fence() -> None:
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_recovers_object_from_prose() -> None:
    """Модель зрідка додає пояснення навколо JSON навіть у json-режимі."""
    assert parse_json_object('Ось результат: {"a": 1} — сподіваюсь, підійде') == {"a": 1}


def test_parse_rejects_non_object() -> None:
    for bad in ("[1, 2]", "просто текст", ""):
        try:
            parse_json_object(bad)
        except ValueError:
            continue
        raise AssertionError(f"мало впасти на {bad!r}")


# --------------------- щаблі 3–4: валідація з першого разу ------------------ #


async def test_valid_response_is_ok(row: InboxRow) -> None:
    client = FakeClient(responses=[payload()])
    result = await extract_one(client, row)

    assert result.meta.status is Status.OK
    assert result.meta.repairs == 0
    assert result.extraction is not None
    assert result.extraction.category is Category.AUTOMATION


async def test_sloppy_but_recoverable_response_costs_no_repair(row: InboxRow) -> None:
    """«Автоматизація» з великої не має коштувати додаткового виклику."""
    client = FakeClient(responses=[payload(category="Автоматизація", priority="HIGH ")])
    result = await extract_one(client, row)

    assert result.meta.status is Status.OK
    assert len(client.calls) == 1


# ------------------------- щабель 5: repair-цикл ---------------------------- #


async def test_invalid_enum_is_repaired(row: InboxRow) -> None:
    client = FakeClient(responses=[payload(category="щось своє"), payload()])
    result = await extract_one(client, row, max_repairs=1)

    assert result.meta.status is Status.REPAIRED
    assert result.meta.repairs == 1
    assert result.extraction is not None
    assert len(client.calls) == 2

    # Модель має отримати назад свою відповідь і конкретну претензію до неї.
    repair_dialog = client.calls[1]
    assert repair_dialog[1][0] == "model"
    assert "category" in repair_dialog[2][1]


async def test_broken_json_is_repaired(row: InboxRow) -> None:
    client = FakeClient(responses=["{ це взагалі не json", payload()])
    result = await extract_one(client, row, max_repairs=1)

    assert result.meta.status is Status.REPAIRED
    assert result.extraction is not None


async def test_semantic_rule_triggers_repair(row: InboxRow) -> None:
    """Прапорець уточнення без питань — теж невалідний вивід, хай і за схемою."""
    client = FakeClient(
        responses=[
            payload(needs_clarification=True, clarification_questions=[]),
            payload(needs_clarification=True, clarification_questions=["Які саме дані?"]),
        ]
    )
    result = await extract_one(client, row, max_repairs=1)

    assert result.meta.status is Status.REPAIRED
    assert result.extraction is not None
    assert result.extraction.clarification_questions == ["Які саме дані?"]


# ------------------- щабель 6: збій без втрати рядка ------------------------ #


async def test_persistently_invalid_output_fails_without_losing_row(row: InboxRow) -> None:
    client = FakeClient(responses=[payload(category="дичина"), payload(category="дичина")])
    result = await extract_one(client, row, max_repairs=1)

    assert result.meta.status is Status.FAILED
    assert result.extraction is None
    assert result.meta.error and "валідацію" in result.meta.error
    # Рядок збережено цілком: дані не втрачені, є з чим піти до людини.
    assert result.id == row.id
    assert result.raw_text == row.raw_text


async def test_repairs_can_be_disabled(row: InboxRow) -> None:
    """max_repairs=0 — режим «не платити за другу спробу»."""
    client = FakeClient(responses=[payload(category="дичина")])
    result = await extract_one(client, row, max_repairs=0)

    assert result.meta.status is Status.FAILED
    assert len(client.calls) == 1


# ------------------ транспортні збої — окремий клас помилок ----------------- #


async def test_transient_error_is_retried(row: InboxRow) -> None:
    client = FakeClient(responses=[TransientLLMError("429"), TransientLLMError("503"), payload()])
    result = await extract_one(client, row, max_retries=3)

    assert result.meta.status is Status.OK
    assert result.meta.attempts == 3
    assert result.meta.repairs == 0  # це не помилка виводу, repair тут ні до чого


async def test_transient_errors_exhaust_retries(row: InboxRow) -> None:
    client = FakeClient(responses=[TransientLLMError("429")] * 5)
    result = await extract_one(client, row, max_retries=2)

    assert result.meta.status is Status.FAILED
    assert result.meta.error and "повтори" in result.meta.error
    assert len(client.calls) == 3  # перша спроба + два повтори


async def test_fatal_error_is_not_retried(row: InboxRow) -> None:
    """Невірний ключ повтором не лікується — не витрачаємо на нього час."""
    client = FakeClient(responses=[FatalLLMError("401 invalid key")] * 3)
    result = await extract_one_safe(client, row, max_retries=3)

    assert result.meta.status is Status.FAILED
    assert len(client.calls) == 1


# ----------------------------- економія викликів ---------------------------- #


async def test_empty_text_skips_the_model() -> None:
    empty = InboxRow(id="REQ-999", channel="Slack", raw_text="   ")
    client = FakeClient(responses=[])
    result = await extract_one(client, empty)

    assert result.meta.status is Status.SKIPPED
    assert client.calls == []  # платити за класифікацію порожнечі нема сенсу
