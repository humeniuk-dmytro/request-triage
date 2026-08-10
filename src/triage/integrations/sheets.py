"""Запис результату в Google Sheet.

Залежності (`gspread`, `google-auth`) лежать в extra `sheets` і імпортуються
всередині функції: без цього прапорця вони не потрібні нікому, а базовий
`uv sync` не має тягнути пів google-cloud заради опційної фічі.

Аркуш перезаписується цілком, а не доповнюється. Прогін тріажу — це знімок
інбоксу на момент часу; дописування створило б таблицю, де той самий запит
лежить у трьох різних станах, і незрозуміло, який із них актуальний.

Про скоупи. Service account — окрема особа з власним порожнім Drive: доступ до
конкретної таблиці дає не скоуп, а шеринг цієї таблиці на його email. Тому за
замовчуванням стоїть рівно `spreadsheets` — ширші права не додають можливостей,
а лише збільшують те, що втрачається разом із файлом ключа. Значення винесене
в `GOOGLE_SCOPES` для випадків, коли потрібне інше (наприклад `drive.file`,
щоб сервіс сам створював таблиці — але тоді вони житимуть у Drive сервісного
акаунта, і їх доведеться шарити назад людині).
"""

from __future__ import annotations

import logging

from ..config import Settings
from ..models import TriagedRequest
from ..report import CSV_COLUMNS, to_rows

log = logging.getLogger(__name__)

WORKSHEET_TITLE = "triage"


def export_to_sheet(settings: Settings, results: list[TriagedRequest]) -> bool:
    if not settings.google_service_account_file or not settings.google_sheet_id:
        log.warning(
            "--sheets вказано, але GOOGLE_SERVICE_ACCOUNT_FILE/GOOGLE_SHEET_ID порожні "
            "— пропускаю. Той самий зріз є в out/output.csv."
        )
        return False

    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError:
        log.error("Не встановлено extra 'sheets'. Виконайте: uv sync --extra sheets")
        return False

    try:
        credentials = Credentials.from_service_account_file(
            settings.google_service_account_file,
            scopes=settings.google_scopes,
        )
        spreadsheet = gspread.authorize(credentials).open_by_key(settings.google_sheet_id)

        try:
            worksheet = spreadsheet.worksheet(WORKSHEET_TITLE)
            worksheet.clear()
        except gspread.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(WORKSHEET_TITLE, rows=100, cols=len(CSV_COLUMNS))

        rows = to_rows(results)
        # Один update замість рядок-за-рядком: інакше 18 запитів — це 19 HTTP-викликів
        # і майже гарантований rate limit на більших обсягах.
        worksheet.update(
            [CSV_COLUMNS] + [[row[column] for column in CSV_COLUMNS] for row in rows],
            "A1",
        )
        worksheet.freeze(rows=1)
    except Exception as exc:  # noqa: BLE001 — доставка не має валити готовий результат
        log.error("не вдалося записати в Google Sheet: %s", exc)
        return False

    log.info("результат записано в Google Sheet (аркуш %s)", WORKSHEET_TITLE)
    return True
