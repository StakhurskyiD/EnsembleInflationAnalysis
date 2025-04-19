#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import gspread
from google.auth import default
from google.auth.transport.requests import AuthorizedSession
from dotenv import load_dotenv
import os

# Автоматично шукає .env у поточному каталозі
load_dotenv()

creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
spreadsheet_id = os.getenv("SPREADSHEET_ID")

# Переконайтеся, що змінні встановлені
if not creds_path or not spreadsheet_id:
    raise RuntimeError("Перевірте, що в .env вказані GOOGLE_APPLICATION_CREDENTIALS та SPREADSHEET_ID")


def fetch_sheet_as_df(spreadsheet_id: str, sheet_name: str) -> pd.DataFrame:
    """
    Підключаємось до Google Sheets через gspread + Application Default Credentials,
    забираємо весь лист і конвертуємо в DataFrame.
    """
    # 1) Отримуємо авторизовану сесію
    creds, _ = default(scopes=['https://www.googleapis.com/auth/spreadsheets.readonly'])
    gc = gspread.authorize(creds)

    # 2) Відкриваємо spreadsheet та лист
    sh = gc.open_by_key(spreadsheet_id)
    worksheet = sh.worksheet(sheet_name)

    # 3) Отримуємо всі значення та конвертуємо в pandas
    data = worksheet.get_all_values()
    df = pd.DataFrame(data[1:], columns=data[0])
    return df

def main():
    # ——————————————————————————————————————————————
    # Налаштування
    MONTHLY_SHEET   = "monthly_data"
    QUARTERLY_SHEET = "quarterly_data"
    # Шляхи для збереження CSV
    OUT_MONTHLY   = "monthly_data.csv"
    OUT_QUARTERLY = "quarterly_data.csv"
    # ——————————————————————————————————————————————

    # Завантажуємо дані
    print(f"Fetching sheet '{MONTHLY_SHEET}' …")
    df_monthly = fetch_sheet_as_df(spreadsheet_id, MONTHLY_SHEET)
    print(f"  → got {len(df_monthly)} rows, saving to {OUT_MONTHLY}")
    print(f"  → got {len(df_monthly.columns)} columns, saving to {OUT_QUARTERLY}")
    df_monthly.to_csv(OUT_MONTHLY, index=False, encoding='utf-8-sig')

    print(f"Fetching sheet '{QUARTERLY_SHEET}' …")
    df_quarterly = fetch_sheet_as_df(spreadsheet_id, QUARTERLY_SHEET)
    print(f"  → got {len(df_quarterly)} rows, saving to {OUT_QUARTERLY}")
    print(f"  → got {len(df_quarterly.columns)} columns, saving to {OUT_QUARTERLY}")
    df_quarterly.to_csv(OUT_QUARTERLY, index=False, encoding='utf-8-sig')

    print("Done.")


if __name__ == "__main__":
    main()