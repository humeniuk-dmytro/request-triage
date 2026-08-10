"""Налаштування прогону. Усе з env, усе з дефолтами, ключі — тільки з оточення."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Прайс за мільйон токенів. Це КОНФІГ, а не константа світу: тарифи змінюються,
# тому вони лежать тут, а не зашиті в коді звіту. На безкоштовному тірі фактична
# вартість дорівнює нулю, і оцінка нижче показує, у що прогін обійшовся б на
# платному. Актуальні цифри: https://ai.google.dev/gemini-api/docs/pricing
PRICING_USD_PER_1M = {
    "gemini-3.5-flash": {"input": 1.50, "output": 9.00},
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
}


def estimate_cost_usd(model: str, prompt_tokens: int, output_tokens: int) -> float | None:
    """Оцінка вартості прогону, або None для моделі, якої немає в таблиці.

    Саме None, а не 0.0: нуль у звіті читається як «безкоштовно», хоча
    насправді означає «ми не знаємо тарифу». Тихо помилкова цифра гірша
    за чесний прочерк.
    """
    price = PRICING_USD_PER_1M.get(model)
    if price is None:
        return None
    return round(
        prompt_tokens / 1_000_000 * price["input"] + output_tokens / 1_000_000 * price["output"],
        6,
    )


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


@dataclass(frozen=True, slots=True)
class Settings:
    api_key: str | None
    model: str
    concurrency: int
    rpm: float
    max_retries: int  # повтори при транспортних збоях (429/5xx/таймаут)
    max_repairs: int  # повтори при невалідному виводі моделі
    telegram_bot_token: str | None
    telegram_chat_id: str | None
    google_service_account_file: str | None
    google_sheet_id: str | None

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv()
        return cls(
            api_key=os.getenv("GEMINI_API_KEY") or None,
            model=os.getenv("TRIAGE_MODEL", "gemini-3.5-flash-lite").strip(),
            concurrency=max(1, _int_env("TRIAGE_CONCURRENCY", 5)),
            # Ліміти безкоштовного тіру дуже різні між моделями: у
            # gemini-2.5-flash це 5 запитів/хв і всього 20 на добу, у
            # 3.5-flash-lite помітно більше. 10 — те, що стабільно проходить
            # на моделі за замовчуванням; ловите 429 — знижуйте.
            # 0 вимикає обмежувач (платний тір).
            rpm=max(0.0, _float_env("TRIAGE_RPM", 10.0)),
            max_retries=max(0, _int_env("TRIAGE_MAX_RETRIES", 3)),
            max_repairs=max(0, _int_env("TRIAGE_MAX_REPAIRS", 1)),
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN") or None,
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID") or None,
            google_service_account_file=os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE") or None,
            google_sheet_id=os.getenv("GOOGLE_SHEET_ID") or None,
        )
