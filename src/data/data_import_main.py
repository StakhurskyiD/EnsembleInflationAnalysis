"""
data_import_main.py

Основний модуль для імпорту та первинного аналізу (перейменування колонок, вивід перших рядків)
місячних, квартальних та річних даних у форматі CSV.

Вміст:
  - Словники мапінгу назв колонок (monthly_columns_map, quarterly_columns_map, yearly_columns_map)
  - Три функції (load_and_map_monthly_data, load_and_map_quarterly_data, load_and_map_yearly_data)
  - Функція main() для прикладу використання
"""

import pandas as pd

from src.data.data_preprocessing.column_mappings import monthly_columns_map, quarterly_columns_map, yearly_columns_map




# ----------------- 1. ФУНКЦІЇ ІМПОРТУ ТА ПЕРЕЙМЕНУВАННЯ ------------------

def load_and_map_monthly_data(file_path: str) -> pd.DataFrame:
    """
    Завантажує CSV із щомісячними даними і перейменовує колонки
    згідно зі словником monthly_columns_map.
    """
    df = pd.read_csv(file_path, sep=',', decimal='.')
    df.rename(columns=monthly_columns_map, inplace=True)
    return df


def load_and_map_quarterly_data(file_path: str) -> pd.DataFrame:
    """
    Завантажує CSV із квартальними даними і перейменовує колонки
    згідно зі словником quarterly_columns_map.
    """
    df = pd.read_csv(file_path, sep=',', decimal='.')
    df.rename(columns=quarterly_columns_map, inplace=True)
    return df


def load_and_map_yearly_data(file_path: str) -> pd.DataFrame:
    """
    Завантажує CSV із річними даними і перейменовує колонки
    згідно зі словником yearly_columns_map.
    """
    df = pd.read_csv(file_path, sep=',', decimal='.')
    df.rename(columns=yearly_columns_map, inplace=True)
    return df


# ----------------- 2. ОСНОВНА ФУНКЦІЯ (ДЕМОНСТРАЦІЯ ВИКОРИСТАННЯ) ------------------

def main():
    """
    Демонструє приклад використання функцій імпорту для
    місячних, квартальних та річних даних.
    """
    # ============= 1. ЗАВАНТАЖЕННЯ МІСЯЧНИХ ДАНИХ =============
    monthly_data_path = "import_data/monthly_data.csv"
    monthly_df = load_and_map_monthly_data(monthly_data_path)
    print("=== МІСЯЧНІ ДАНІ ===")
    print("Колонки після мапінгу:", monthly_df.columns.tolist())
    print("Перші 3 рядки:\n", monthly_df.head(3), "\n")

    # ============= 2. ЗАВАНТАЖЕННЯ КВАРТАЛЬНИХ ДАНИХ =============
    quarterly_data_path = "import_data/quarterly_data.csv"
    quarterly_df = load_and_map_quarterly_data(quarterly_data_path)
    print("=== КВАРТАЛЬНІ ДАНІ ===")
    print("Колонки після мапінгу:", quarterly_df.columns.tolist())
    print("Перші 3 рядки:\n", quarterly_df.head(3), "\n")

    # ============= 3. ЗАВАНТАЖЕННЯ РІЧНИХ ДАНИХ =============
    yearly_data_path = "import_data/annual_data.csv"
    yearly_df = load_and_map_yearly_data(yearly_data_path)
    print("=== РІЧНІ ДАНІ ===")
    print("Колонки після мапінгу:", yearly_df.columns.tolist())
    print("Перші 3 рядки:\n", yearly_df.head(3), "\n")


# ----------------- 3. ЗАПУСК СКРИПТА ------------------
if __name__ == "__main__":
    main()