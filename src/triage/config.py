"""Налаштування прогону. Усе з env, усе з дефолтами, ключі — тільки з оточення."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Прайс за мільйон токенів. Це КОНФІГ, а не константа світу: тарифи змінюються,
# тому число лежить тут, а не зашите в коді звіту. На безкоштовному тірі
# Gemini фактична вартість = 0, і оцінка нижче показує, у що прогін обійшовся б
# на платному. Актуальні цифри: https://ai.google.dev/pricing
PRICING_USD_PER_1M = {
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
}
_FALLBACK_PRICE = {"input": 0.0, "output": 0.0}


def estimate_cost_usd(model: str, prompt_tokens: int, output_tokens: int) -> float:
    """Оцінка вартості прогону. Невідома модель → 0.0 і чесний нуль у звіті."""
    price = PRICING_USD_PER_1M.get(model, _FALLBACK_PRICE)
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
            model=os.getenv("TRIAGE_MODEL", "gemini-2.5-flash").strip(),
            concurrency=max(1, _int_env("TRIAGE_CONCURRENCY", 5)),
            # Безкоштовний тір gemini-2.5-flash дозволяє 5 запитів/хв. Тримаємо
            # 4 — запас у 20%: рівно на межі провайдер усе одно віддає 429, бо
            # рахує у власних вікнах, не синхронних із нашими.
            # 0 вимикає обмежувач (платний тір або інша модель).
            rpm=max(0.0, _float_env("TRIAGE_RPM", 4.0)),
            max_retries=max(0, _int_env("TRIAGE_MAX_RETRIES", 3)),
            max_repairs=max(0, _int_env("TRIAGE_MAX_REPAIRS", 1)),
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN") or None,
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID") or None,
            google_service_account_file=os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE") or None,
            google_sheet_id=os.getenv("GOOGLE_SHEET_ID") or None,
        )
