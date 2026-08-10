"""Налаштування й оцінка вартості."""

from __future__ import annotations

import pytest

from triage.config import DEFAULT_GOOGLE_SCOPE, PRICING_USD_PER_1M, Settings, estimate_cost_usd
from triage.report import format_cost


def test_cost_is_computed_from_the_price_table() -> None:
    price = PRICING_USD_PER_1M["gemini-2.5-flash"]
    cost = estimate_cost_usd("gemini-2.5-flash", 1_000_000, 1_000_000)
    assert cost == pytest.approx(price["input"] + price["output"])


def test_unknown_model_reports_no_estimate_instead_of_zero() -> None:
    """Нуль читається як «безкоштовно» — це була б тиха брехня у звіті."""
    assert estimate_cost_usd("щось-нове-2027", 10_000, 1_000) is None
    assert format_cost(None) == "оцінка недоступна"


def test_settings_read_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("TRIAGE_RPM", "12")
    monkeypatch.setenv("TRIAGE_CONCURRENCY", "3")

    settings = Settings.from_env()

    assert settings.api_key == "k"
    assert settings.rpm == 12
    assert settings.concurrency == 3


def test_broken_env_values_fall_back_to_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Одруківка в .env не має валити прогін мовчазним TypeError."""
    monkeypatch.setenv("TRIAGE_RPM", "швидко")
    monkeypatch.setenv("TRIAGE_CONCURRENCY", "")

    settings = Settings.from_env()

    assert settings.rpm > 0
    assert settings.concurrency >= 1


def test_google_scope_is_narrow_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ключ у файлі має відкривати один аркуш, а не пів-акаунта."""
    monkeypatch.delenv("GOOGLE_SCOPES", raising=False)
    assert Settings.from_env().google_scopes == (DEFAULT_GOOGLE_SCOPE,)


def test_google_scopes_are_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_SCOPES", "https://a/scope1, https://a/scope2")
    assert Settings.from_env().google_scopes == ("https://a/scope1", "https://a/scope2")
