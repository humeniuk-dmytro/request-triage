"""Схема: нормалізація, семантичні правила, синхронність із схемою для API."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from triage.models import (
    GEMINI_RESPONSE_SCHEMA,
    Category,
    Confidence,
    Extraction,
    Priority,
)

from .conftest import VALID_PAYLOAD


def test_schema_matches_model() -> None:
    """Схема для Gemini написана руками — цей тест не дає їй розійтися з моделлю.

    Розбіжність не помітна на око: модель просто перестає отримувати обмеження
    на поле, яке ми додали, і починає його вигадувати.
    """
    schema_fields = set(GEMINI_RESPONSE_SCHEMA["properties"])
    assert schema_fields == set(Extraction.model_fields)
    assert set(GEMINI_RESPONSE_SCHEMA["required"]) == schema_fields
    assert set(GEMINI_RESPONSE_SCHEMA["propertyOrdering"]) == schema_fields
    # reasoning генерується першим — на цьому тримається якість класифікації.
    assert GEMINI_RESPONSE_SCHEMA["propertyOrdering"][0] == "reasoning"


def test_enum_values_match_spec() -> None:
    """Значення категорій — дослівно з ТЗ. Перейменування тут ламає замовника."""
    assert [c.value for c in Category] == [
        "автоматизація",
        "інтеграція",
        "звіт/аналітика",
        "баг/підтримка",
        "питання/консультація",
        "поза скоупом",
    ]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Автоматизація", Category.AUTOMATION),
        ("  автоматизація  ", Category.AUTOMATION),
        ("звіт / аналітика", Category.ANALYTICS),
        ("Звіт", Category.ANALYTICS),
        ("аналітика", Category.ANALYTICS),
        ("BUG", Category.SUPPORT),
        ("out of scope", Category.OUT_OF_SCOPE),
    ],
)
def test_category_normalisation(raw: str, expected: Category) -> None:
    """Валідне-але-неохайне має проходити без платного repair-виклику."""
    assert Extraction.model_validate({**VALID_PAYLOAD, "category": raw}).category is expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("HIGH ", Priority.HIGH),
        ("високий", Priority.HIGH),
        ("urgent", Priority.HIGH),
        ("Medium", Priority.MEDIUM),
        ("низька", Priority.LOW),
    ],
)
def test_priority_normalisation(raw: str, expected: Priority) -> None:
    assert Extraction.model_validate({**VALID_PAYLOAD, "priority": raw}).priority is expected


def test_string_instead_of_list_is_wrapped() -> None:
    e = Extraction.model_validate({**VALID_PAYLOAD, "requested_actions": "зробити звіт"})
    assert e.requested_actions == ["зробити звіт"]


def test_null_placeholders_become_none() -> None:
    """«невідомо» — це не назва відділу."""
    for placeholder in ("невідомо", "", "-", "не вказано"):
        e = Extraction.model_validate({**VALID_PAYLOAD, "target_department": placeholder})
        assert e.target_department is None


def test_string_bool_is_coerced() -> None:
    e = Extraction.model_validate(
        {**VALID_PAYLOAD, "needs_clarification": "так", "clarification_questions": ["що саме?"]}
    )
    assert e.needs_clarification is True


def test_none_list_becomes_empty() -> None:
    assert (
        Extraction.model_validate({**VALID_PAYLOAD, "mentioned_systems": None}).mentioned_systems
        == []
    )


def test_unknown_category_is_rejected() -> None:
    """Нормалізація виправляє форму, а не вигадує значення поза довідником."""
    with pytest.raises(ValidationError):
        Extraction.model_validate({**VALID_PAYLOAD, "category": "щось своє"})


def test_extra_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Extraction.model_validate({**VALID_PAYLOAD, "urgency_score": 9000})


def test_clarification_flag_requires_questions() -> None:
    """Прапорець без питань — робота, яку модель не доробила."""
    with pytest.raises(ValidationError, match="clarification_questions"):
        Extraction.model_validate(
            {**VALID_PAYLOAD, "needs_clarification": True, "clarification_questions": []}
        )


def test_questions_require_clarification_flag() -> None:
    with pytest.raises(ValidationError, match="визначся"):
        Extraction.model_validate(
            {**VALID_PAYLOAD, "needs_clarification": False, "clarification_questions": ["а що?"]}
        )


def test_empty_summary_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Extraction.model_validate({**VALID_PAYLOAD, "short_summary": ""})


def test_confidence_defaults_to_medium() -> None:
    data = {k: v for k, v in VALID_PAYLOAD.items() if k != "confidence"}
    assert Extraction.model_validate(data).confidence is Confidence.MEDIUM
