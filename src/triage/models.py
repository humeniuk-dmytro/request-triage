"""Контракт даних: що приходить на вхід, що зобов'язана повернути модель, що йде на вихід.

Тут єдине джерело правди для схеми. LLM обмежена схемою ще на рівні API
(див. GEMINI_RESPONSE_SCHEMA), але вивід усе одно проходить повну валідацію
тут — модель може віддати синтаксично валідний JSON із семантичним сміттям.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

# --------------------------------------------------------------------------- #
# Довідники
# --------------------------------------------------------------------------- #


class Category(StrEnum):
    """Шість категорій із ТЗ, дослівно. Значення — те, що потрапить у вивід."""

    AUTOMATION = "автоматизація"
    INTEGRATION = "інтеграція"
    ANALYTICS = "звіт/аналітика"
    SUPPORT = "баг/підтримка"
    CONSULTATION = "питання/консультація"
    OUT_OF_SCOPE = "поза скоупом"


class Priority(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Confidence(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Status(StrEnum):
    """Що сталося із запитом під час обробки — видно в кожному записі виводу."""

    OK = "ok"  # провалідовано з першої спроби
    REPAIRED = "repaired"  # модель віддала невалідне, виправила після зауваження
    FAILED = "failed"  # не вдалося отримати валідний вивід; рядок збережено
    SKIPPED = "skipped"  # порожній raw_text — LLM не викликали, щоб не палити токени


# --------------------------------------------------------------------------- #
# Нормалізація до валідації
#
# Моделі регулярно віддають "валідне, але неохайне": "Автоматизація" з великої,
# "HIGH " з пробілом, рядок замість списку, "так" замість true. Кожен такий
# випадок, доведений до ValidationError, коштує зайвий платний repair-виклик.
# Дешевше нормалізувати те, що однозначно відновлюється.
# --------------------------------------------------------------------------- #

_NULLISH = {"", "-", "—", "null", "none", "n/a", "невідомо", "не вказано", "не зрозуміло"}

_CATEGORY_SYNONYMS = {
    "автоматизації": "автоматизація",
    "автоматизацію": "автоматизація",
    "automation": "автоматизація",
    "інтеграції": "інтеграція",
    "інтеграцію": "інтеграція",
    "integration": "інтеграція",
    "звіт": "звіт/аналітика",
    "звіти": "звіт/аналітика",
    "звітність": "звіт/аналітика",
    "аналітика": "звіт/аналітика",
    "звіт та аналітика": "звіт/аналітика",
    "report": "звіт/аналітика",
    "analytics": "звіт/аналітика",
    "баг": "баг/підтримка",
    "підтримка": "баг/підтримка",
    "баг та підтримка": "баг/підтримка",
    "bug": "баг/підтримка",
    "support": "баг/підтримка",
    "питання": "питання/консультація",
    "консультація": "питання/консультація",
    "question": "питання/консультація",
    "consultation": "питання/консультація",
    "поза скоуп": "поза скоупом",
    "out of scope": "поза скоупом",
    "out_of_scope": "поза скоупом",
}

_LEVEL_SYNONYMS = {
    "низький": "low",
    "низька": "low",
    "середній": "medium",
    "середня": "medium",
    "високий": "high",
    "висока": "high",
    "терміново": "high",
    "urgent": "high",
    "critical": "high",
    "normal": "medium",
}


def _norm_enum(value: Any, synonyms: dict[str, str]) -> Any:
    """Нижній регістр, стиснуті пробіли, розкриття синонімів."""
    if not isinstance(value, str):
        return value
    v = re.sub(r"\s*/\s*", "/", " ".join(value.split())).strip().lower()
    return synonyms.get(v, v)


def _norm_category(value: Any) -> Any:
    return _norm_enum(value, _CATEGORY_SYNONYMS)


def _norm_level(value: Any) -> Any:
    """Спільний нормалізатор для шкал low/medium/high (priority і confidence)."""
    return _norm_enum(value, _LEVEL_SYNONYMS)


def _norm_optional_str(value: Any) -> Any:
    """Порожні заглушки моделі ('невідомо', '-', '') — це null, а не значення."""
    if value is None:
        return None
    if isinstance(value, str):
        v = " ".join(value.split())
        return None if v.lower() in _NULLISH else v
    return value


def _norm_str_list(value: Any) -> Any:
    """None → [], одиночний рядок → [рядок], список → очищений список рядків."""
    if value is None:
        return []
    if isinstance(value, str):
        v = " ".join(value.split())
        return [] if v.lower() in _NULLISH else [v]
    if isinstance(value, list):
        out = []
        for item in value:
            if item is None:
                continue
            s = " ".join(str(item).split())
            if s and s.lower() not in _NULLISH:
                out.append(s)
        return out
    return value


def _norm_bool(value: Any) -> Any:
    """Моделі часто відповідають 'true'/'так' рядком замість булевого."""
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "yes", "так", "1"}:
            return True
        if v in {"false", "no", "ні", "0"}:
            return False
    return value


NormCategory = Annotated[Category, BeforeValidator(_norm_category)]
NormPriority = Annotated[Priority, BeforeValidator(_norm_level)]
NormConfidence = Annotated[Confidence, BeforeValidator(_norm_level)]
OptionalStr = Annotated[str | None, BeforeValidator(_norm_optional_str)]
StrList = Annotated[list[str], BeforeValidator(_norm_str_list)]
NormBool = Annotated[bool, BeforeValidator(_norm_bool)]


# --------------------------------------------------------------------------- #
# Вхід
# --------------------------------------------------------------------------- #


class InboxRow(BaseModel):
    """Рядок вивантаження інбоксу — рівно поля з input_requests.csv."""

    model_config = ConfigDict(str_strip_whitespace=True)

    id: str = Field(min_length=1)
    channel: str = ""
    # Порожня клітинка часу — норма для реального вивантаження, а не привід
    # викинути рядок: текст запиту в ньому нікуди не подівся.
    timestamp: Annotated[datetime | None, BeforeValidator(_norm_optional_str)] = None
    raw_text: str = ""


# --------------------------------------------------------------------------- #
# Те, що зобов'язана повернути модель
# --------------------------------------------------------------------------- #


class Extraction(BaseModel):
    """Строга схема виводу LLM.

    extra="forbid" — свідомо. Зайве поле означає, що модель імпровізує замість
    того, щоб дотримуватись контракту; це сигнал деградації, а не нешкідливий
    шум. Ціна рішення — можливий зайвий repair-виклик, і вона прийнятна.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    # Поле стоїть першим навмисно: у propertyOrdering схеми Gemini воно теж
    # перше, тобто модель формулює аргумент ДО того, як обере категорію.
    # Це дешевий chain-of-thought усередині структурованого виводу —
    # без окремого reasoning-виклику й без увімкненого thinking.
    reasoning: str = Field(min_length=1, max_length=500)

    # --- обов'язкові поля з ТЗ ---
    category: NormCategory
    target_department: OptionalStr = None
    priority: NormPriority
    short_summary: str = Field(min_length=1, max_length=400)
    requested_actions: StrList = []
    needs_clarification: NormBool = False

    # --- розширення схеми (обґрунтування — у README) ---
    clarification_questions: StrList = []
    confidence: NormConfidence = Confidence.MEDIUM
    mentioned_systems: StrList = []

    @model_validator(mode="after")
    def _clarification_needs_questions(self) -> Extraction:
        """Прапорець 'потребує уточнення' без самих питань — марний.

        Це семантична перевірка, якої не зробить жодна JSON-схема: вона ловить
        модель, що поставила прапорець і не виконала роботу. Порушення йде
        в repair-цикл із цим самим текстом помилки.
        """
        if self.needs_clarification and not self.clarification_questions:
            raise ValueError(
                "needs_clarification=true вимагає щонайменше одного питання "
                "в clarification_questions"
            )
        if not self.needs_clarification and self.clarification_questions:
            raise ValueError(
                "clarification_questions заповнено, але needs_clarification=false — "
                "визначся: або запит зрозумілий, або потребує уточнення"
            )
        return self


# --------------------------------------------------------------------------- #
# Схема для API Gemini
#
# Пишемо руками, а не через model_json_schema(): Gemini приймає лише підмножину
# JSON Schema (без $ref/$defs/anyOf, типи великими літерами). Pydantic лишається
# джерелом правди для валідації, а ця схема — обмеженням на рівні API.
# Тест test_models.py::test_schema_matches_model стежить, щоб вони не розійшлися.
# --------------------------------------------------------------------------- #

GEMINI_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "required": [
        "reasoning",
        "category",
        "target_department",
        "priority",
        "short_summary",
        "requested_actions",
        "needs_clarification",
        "clarification_questions",
        "confidence",
        "mentioned_systems",
    ],
    # Порядок генерації = порядок міркування. reasoning першим — навмисно.
    "propertyOrdering": [
        "reasoning",
        "category",
        "priority",
        "target_department",
        "short_summary",
        "requested_actions",
        "needs_clarification",
        "clarification_questions",
        "confidence",
        "mentioned_systems",
    ],
    "properties": {
        "reasoning": {
            "type": "STRING",
            "description": "Одне речення: чому саме ця категорія і цей пріоритет.",
        },
        "category": {"type": "STRING", "enum": [c.value for c in Category]},
        "priority": {"type": "STRING", "enum": [p.value for p in Priority]},
        "target_department": {
            "type": "STRING",
            "nullable": True,
            "description": "Відділ-замовник; null, якщо з тексту не визначається.",
        },
        "short_summary": {"type": "STRING", "description": "Суть запиту одним реченням."},
        "requested_actions": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "Конкретні дії, які просять зробити. Може бути порожнім.",
        },
        "needs_clarification": {"type": "BOOLEAN"},
        "clarification_questions": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "Питання замовнику. Непорожній ⟺ needs_clarification=true.",
        },
        "confidence": {"type": "STRING", "enum": [c.value for c in Confidence]},
        "mentioned_systems": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "Згадані системи та сервіси: Google Ads, PlanFix, BigQuery…",
        },
    },
}


# --------------------------------------------------------------------------- #
# Вихід
# --------------------------------------------------------------------------- #


class ProcessingMeta(BaseModel):
    """Технічний слід обробки. Дає відповідь на 'чому цей запис такий'."""

    status: Status
    attempts: int = 0  # звернень до API, разом із повторами після 429/5xx
    repairs: int = 0  # скільки разів просили виправити невалідний вивід
    model: str = ""
    prompt_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    error: str | None = None


class TriagedRequest(BaseModel):
    """Один рядок інбоксу після обробки.

    Вхідні поля лишаються поряд із витягнутими навмисно: рев'юер (і людина
    на чергуванні) бачить першоджерело поруч із висновком моделі.
    """

    id: str
    channel: str = ""
    timestamp: datetime | None = None
    raw_text: str = ""

    extraction: Extraction | None = None
    # Заповнюється окремим проходом (dedupe.py): у порядковому виклику модель
    # не бачить інших рядків, тому дублі принципово не визначаються там.
    possible_duplicate_of: str | None = None
    meta: ProcessingMeta


class Aggregates(BaseModel):
    """Агрегати для звіту — рахуються кодом, не моделлю."""

    total: int
    by_category: dict[str, int]
    by_priority: dict[str, int]
    by_department: dict[str, int]
    by_channel: dict[str, int]
    by_status: dict[str, int]
    by_confidence: dict[str, int]
    needs_clarification_ids: list[str]
    duplicate_pairs: list[tuple[str, str]]
    failed_ids: list[str]


class RunManifest(BaseModel):
    """Паспорт прогону: без нього результат неможливо відтворити чи перевірити."""

    started_at: datetime
    finished_at: datetime
    provider: Literal["gemini", "fake"]
    model: str
    prompt_sha256: str  # змінився промпт — змінився і хеш, розбіжність пояснена
    input_file: str
    requests_total: int
    concurrency: int
    prompt_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    wall_time_sec: float


class RunResult(BaseModel):
    """Те, що серіалізується в output.json."""

    manifest: RunManifest
    aggregates: Aggregates
    requests: list[TriagedRequest]
