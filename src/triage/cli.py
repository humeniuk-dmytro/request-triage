"""Точка входу: `uv run triage` або `python -m triage`."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from .config import Settings, estimate_cost_usd
from .dedupe import DuplicatePair, find_duplicates
from .llm import build_client
from .llm.base import FatalLLMError
from .models import RunManifest, RunResult, Status, TriagedRequest
from .pipeline import read_inbox, run_triage
from .prompt import prompt_fingerprint
from .report import compute_aggregates, format_cost, write_outputs

log = logging.getLogger("triage")

DEFAULT_INPUT = Path("data/input_requests.csv")
DEFAULT_OUT = Path("out")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="triage",
        description="Класифікація вхідних запитів AI-юніту через LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="CSV із запитами")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="тека для результатів")
    p.add_argument(
        "--provider",
        choices=["gemini", "fake"],
        default="gemini",
        help="fake — детермінована евристика без мережі й без ключа",
    )
    p.add_argument("--model", default=None, help="перевизначити модель з .env")
    p.add_argument("--concurrency", type=int, default=None, help="паралельних запитів")
    p.add_argument(
        "--rpm",
        type=float,
        default=None,
        help="ліміт запитів на хвилину (0 — без обмеження)",
    )
    p.add_argument("--limit", type=int, default=None, help="обробити лише перші N рядків")
    p.add_argument("--no-dedupe", action="store_true", help="пропустити прохід по дублях")
    p.add_argument("--notify", action="store_true", help="надіслати дайджест у Telegram")
    p.add_argument("--sheets", action="store_true", help="записати результат у Google Sheet")
    p.add_argument("-v", "--verbose", action="store_true", help="докладний лог")
    return p


async def _run(args: argparse.Namespace, settings: Settings) -> int:
    rows, problems = read_inbox(args.input)
    for problem in problems:
        log.warning("вхідні дані: %s", problem)
    if not rows:
        log.error("У %s немає жодного придатного рядка.", args.input)
        return 1
    if args.limit:
        rows = rows[: args.limit]

    client = build_client(args.provider, settings)
    concurrency = args.concurrency or settings.concurrency
    log.info(
        "обробляю %d запитів: провайдер=%s модель=%s паралельність=%d ліміт=%.0f/хв",
        len(rows),
        client.name,
        client.model,
        concurrency,
        settings.rpm,
    )

    started_at = datetime.now()
    clock = time.perf_counter()
    done_count = 0

    def on_done(result: TriagedRequest) -> None:
        nonlocal done_count
        done_count += 1
        marker = {Status.OK: "·", Status.REPAIRED: "~", Status.SKIPPED: "-"}.get(
            result.meta.status, "!"
        )
        category = result.extraction.category.value if result.extraction else "—"
        log.info("[%2d/%2d] %s %s → %s", done_count, len(rows), marker, result.id, category)

    try:
        results = await run_triage(
            client,
            rows,
            concurrency=concurrency,
            max_retries=settings.max_retries,
            max_repairs=settings.max_repairs,
            on_done=on_done,
        )

        duplicates: list[DuplicatePair] = []
        dedupe_in = dedupe_out = 0
        if not args.no_dedupe:
            duplicates, dedupe_in, dedupe_out = await find_duplicates(
                client, results, max_retries=settings.max_retries
            )
    finally:
        await client.aclose()

    prompt_tokens = sum(r.meta.prompt_tokens for r in results) + dedupe_in
    output_tokens = sum(r.meta.output_tokens for r in results) + dedupe_out

    manifest = RunManifest(
        started_at=started_at,
        finished_at=datetime.now(),
        provider=args.provider,
        model=client.model,
        prompt_sha256=prompt_fingerprint(),
        input_file=str(args.input),
        requests_total=len(results),
        concurrency=concurrency,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        total_tokens=prompt_tokens + output_tokens,
        estimated_cost_usd=estimate_cost_usd(client.model, prompt_tokens, output_tokens),
        wall_time_sec=round(time.perf_counter() - clock, 2),
    )
    result = RunResult(
        manifest=manifest,
        aggregates=compute_aggregates(results, duplicates),
        requests=results,
    )
    paths = write_outputs(args.out, result, duplicates)

    # Доставка результату не має права зіпсувати сам результат: він уже на диску.
    if args.sheets:
        from .integrations.sheets import export_to_sheet

        export_to_sheet(settings, result.requests)
    if args.notify:
        from .integrations.telegram import send_digest

        await send_digest(settings, result, duplicates)

    agg = result.aggregates
    print()
    print(
        f"Оброблено: {agg.total}  ·  {manifest.wall_time_sec:.1f} с  ·  "
        f"{manifest.total_tokens} токенів  ·  {format_cost(manifest.estimated_cost_usd)}"
    )
    print(f"Статуси: {agg.by_status}")
    print(
        f"Потребують уточнення: {len(agg.needs_clarification_ids)}  ·  "
        f"можливих дублів: {len(agg.duplicate_pairs)}  ·  "
        f"збоїв: {len(agg.failed_ids)}"
    )
    print(f"Файли: {paths['json']}, {paths['report']}, {paths['csv']}")

    # Ненульовий код, якщо жоден запит не вдалось обробити: у CI/крон це має
    # бути помітно, а не «зелений» прогін із порожнім звітом.
    return 0 if agg.total > len(agg.failed_ids) else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(message)s",
        stream=sys.stderr,
    )
    if not args.verbose:
        # SDK і http-клієнт логують кожен виклик на INFO; у звичайному прогоні
        # це ховає власні повідомлення про хід обробки. З -v усе повертається.
        for noisy in ("google_genai", "httpx", "httpcore", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    settings = Settings.from_env()
    overrides = {}
    if args.model:
        overrides["model"] = args.model
    if args.rpm is not None:
        overrides["rpm"] = args.rpm
    if overrides:
        # dataclasses.replace, а не __dict__: Settings — frozen зі slots.
        settings = dataclasses.replace(settings, **overrides)

    if args.provider == "gemini" and not settings.api_key:
        log.error(
            "Немає GEMINI_API_KEY (покладіть у .env). Безкоштовний ключ: "
            "https://aistudio.google.com/apikey\n"
            "        Щоб побачити роботу конвеєра без ключа: --provider fake"
        )
        return 2

    try:
        return asyncio.run(_run(args, settings))
    except (FileNotFoundError, ValueError, FatalLLMError) as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.warning("перервано користувачем")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
