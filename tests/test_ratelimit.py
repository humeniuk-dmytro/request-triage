"""Темп звернень і поведінка на 429.

Ці тести з'явилися після реального прогону: безкоштовний тір Gemini дає
5 запитів на хвилину, і перша версія в нього гарантовано впиралась.
"""

from __future__ import annotations

import asyncio

from triage.extractor import call_with_retry
from triage.llm.base import TransientLLMError
from triage.llm.fake import FakeClient
from triage.llm.ratelimit import RateLimitedClient, RateLimiter
from triage.models import FailureKind, InboxRow, Status
from triage.pipeline import run_triage

from .conftest import payload


async def test_limiter_spaces_calls_evenly(real_sleep: None) -> None:
    """60/rpm між стартами: три виклики при 120/хв — це щонайменше секунда."""
    limiter = RateLimiter(rpm=120)  # 0.5 с між стартами
    loop = asyncio.get_running_loop()

    start = loop.time()
    await asyncio.gather(*(limiter.acquire() for _ in range(3)))
    elapsed = loop.time() - start

    assert elapsed >= 1.0


async def test_limiter_can_be_disabled(real_sleep: None) -> None:
    limiter = RateLimiter(rpm=0)
    loop = asyncio.get_running_loop()

    start = loop.time()
    await asyncio.gather(*(limiter.acquire() for _ in range(20)))

    assert loop.time() - start < 0.1


async def test_wrapper_keeps_the_client_contract() -> None:
    inner = FakeClient(responses=[payload()])
    client = RateLimitedClient(inner, rpm=0)

    assert client.model == inner.model
    response = await client.complete(system="s", messages=[("user", "u")], schema={})
    assert response.text
    assert len(inner.calls) == 1


async def test_server_retry_delay_beats_our_guess(sleeps: list[float]) -> None:
    """Коли сервер каже «retry in 28s», спати 2 секунди — це ще один 429."""
    client = FakeClient(responses=[TransientLLMError("429 quota", retry_after=28.0), payload()])
    await call_with_retry(client, system="s", messages=[("user", "u")], schema={}, max_retries=2)

    assert sleeps and 28.0 <= sleeps[0] <= 29.0


async def test_exponential_backoff_when_server_is_silent(sleeps: list[float]) -> None:
    client = FakeClient(responses=[TransientLLMError("503"), TransientLLMError("503"), payload()])
    await call_with_retry(client, system="s", messages=[("user", "u")], schema={}, max_retries=3)

    assert len(sleeps) == 2
    assert sleeps[1] > sleeps[0]  # росте


async def test_rate_limit_does_not_trip_the_breaker() -> None:
    """Вичерпаний ліміт — не зламаний конфіг.

    Саме на цьому спіткнувся перший реальний прогін: запобіжник порахував
    серію 429 за поламаний ключ і зупинив обробку решти файлу.
    """
    rows = [InboxRow(id=f"REQ-{i}", raw_text=f"запит {i}") for i in range(1, 7)]
    client = FakeClient(responses=[TransientLLMError("429 quota")] * 100)

    results = await run_triage(client, rows, concurrency=1, max_retries=1, breaker_threshold=3)

    assert all(r.meta.status is Status.FAILED for r in results)
    assert all(r.meta.failure_kind is FailureKind.TRANSIENT for r in results)
    # Жоден рядок не має бути зупинений запобіжником.
    assert not any("запобіжником" in (r.meta.error or "") for r in results)
