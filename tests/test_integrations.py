"""Опційні виходи: Telegram і Google Sheets.

Головне, що тут перевіряється — не «фіча працює», а «фіча не має права
завалити прогін». Результат уже на диску до того, як хтось спробує його
доставити, тому будь-яка проблема з доставкою — це попередження, а не збій.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime

import pytest

from triage.config import DEFAULT_GOOGLE_SCOPE, Settings
from triage.dedupe import DuplicatePair
from triage.integrations.sheets import export_to_sheet
from triage.integrations.telegram import TELEGRAM_LIMIT, render_digest, send_digest
from triage.models import (
    Extraction,
    ProcessingMeta,
    RunManifest,
    RunResult,
    Status,
    TriagedRequest,
)
from triage.report import compute_aggregates

from .conftest import VALID_PAYLOAD


def _empty_settings() -> Settings:
    return Settings(
        api_key=None,
        model="test",
        concurrency=1,
        rpm=0,
        max_retries=0,
        max_repairs=0,
        telegram_bot_token=None,
        telegram_chat_id=None,
        google_service_account_file=None,
        google_sheet_id=None,
        google_scopes=(DEFAULT_GOOGLE_SCOPE,),
    )


def _result(requests: list[TriagedRequest], duplicates: list[DuplicatePair]) -> RunResult:
    now = datetime(2026, 6, 9, 10, 0, 0)
    return RunResult(
        manifest=RunManifest(
            started_at=now,
            finished_at=now,
            provider="fake",
            model="test",
            prompt_sha256="abc123",
            input_file="data/input_requests.csv",
            requests_total=len(requests),
            concurrency=1,
            prompt_tokens=10,
            output_tokens=5,
            total_tokens=15,
            estimated_cost_usd=0.0001,
            wall_time_sec=1.0,
        ),
        aggregates=compute_aggregates(requests, duplicates),
        requests=requests,
    )


def _req(rid: str, **fields) -> TriagedRequest:
    return TriagedRequest(
        id=rid,
        channel="Slack",
        timestamp="2026-06-08 09:14",
        raw_text="текст",
        extraction=Extraction.model_validate({**VALID_PAYLOAD, **fields}),
        meta=ProcessingMeta(status=Status.OK, model="test"),
    )


# ------------------------------- Telegram ---------------------------------- #


def test_digest_leads_with_what_needs_action_today() -> None:
    requests = [
        _req("REQ-005", priority="high", short_summary="Терміново вивантажити контрагентів"),
        _req("REQ-004", priority="low"),
    ]
    digest = render_digest(_result(requests, []), [])

    assert "Сьогодні (high)" in digest
    assert "REQ-005" in digest
    # Низький пріоритет не має засмічувати дайджест поіменно.
    assert digest.index("Сьогодні") < digest.index("REQ-005")


def test_digest_lists_clarifications_and_duplicates() -> None:
    requests = [
        _req("REQ-001"),
        _req(
            "REQ-013",
            needs_clarification=True,
            clarification_questions=["Які саме дані?"],
        ),
    ]
    duplicates = [DuplicatePair(duplicate_id="REQ-013", original_id="REQ-001", reason="те саме")]
    digest = render_digest(_result(requests, duplicates), duplicates)

    assert "Потребують уточнення" in digest and "REQ-013" in digest
    assert "REQ-013 ≈ REQ-001" in digest


def test_digest_is_trimmed_to_telegram_limit() -> None:
    """4096 символів — жорсткий ліміт API; довше просто не відправиться."""
    requests = [
        _req(f"REQ-{i:03d}", priority="high", short_summary="дуже довга суть запиту " * 12)
        for i in range(1, 60)
    ]
    digest = render_digest(_result(requests, []), [])

    assert len(digest) <= TELEGRAM_LIMIT


async def test_notify_without_token_is_skipped_not_failed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    result = _result([_req("REQ-001")], [])

    assert await send_digest(_empty_settings(), result, []) is False
    assert "пропускаю" in caplog.text


async def test_notify_survives_a_dead_telegram(monkeypatch: pytest.MonkeyPatch) -> None:
    """Впав Telegram — прогін усе одно вдалий: результат давно на диску."""
    import httpx

    settings = dataclasses.replace(
        _empty_settings(), telegram_bot_token="123:abc", telegram_chat_id="42"
    )

    class DeadClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def post(self, *_args, **_kwargs):
            raise httpx.ConnectError("немає мережі")

    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: DeadClient())

    assert await send_digest(settings, _result([_req("REQ-001")], []), []) is False


# ------------------------------ Google Sheets ------------------------------- #


def test_sheets_without_credentials_is_skipped_not_failed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert export_to_sheet(_empty_settings(), [_req("REQ-001")]) is False
    assert "пропускаю" in caplog.text
    # Порада, куди дивитись замість цього, важливіша за сам факт пропуску.
    assert "output.csv" in caplog.text


def test_sheets_export_uses_the_configured_scopes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Скоуп не має тихо розширюватись повз налаштування."""
    captured: dict[str, object] = {}

    settings = dataclasses.replace(
        _empty_settings(),
        google_service_account_file="/tmp/sa.json",
        google_sheet_id="sheet-1",
        google_scopes=("https://a/only-this",),
    )

    import triage.integrations.sheets as sheets_module

    class FakeCredentials:
        @staticmethod
        def from_service_account_file(path, scopes):
            captured["path"] = path
            captured["scopes"] = scopes
            raise RuntimeError("далі не йдемо — нас цікавлять лише скоупи")

    monkeypatch.setitem(
        __import__("sys").modules,
        "google.oauth2.service_account",
        type("M", (), {"Credentials": FakeCredentials}),
    )
    monkeypatch.setitem(__import__("sys").modules, "gspread", type("M", (), {}))

    assert sheets_module.export_to_sheet(settings, [_req("REQ-001")]) is False
    assert captured["scopes"] == ("https://a/only-this",)
