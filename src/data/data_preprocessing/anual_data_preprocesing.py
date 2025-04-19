import pandas as pd
import numpy as np
import os
import re
from statsmodels.tsa.stattools import adfuller
from src.data.data_preprocessing.column_mappings import yearly_columns_map


def load_and_map_annual_data(file_path: str) -> pd.DataFrame:
    """
    Завантажує CSV з річними даними і перейменовує колонки
    згідно зі словником yearly_columns_map.
    """
    # Якщо CSV використовує десятковий роздільник як '.', можна залишити decimal='.'.
    # Якщо ж у даних використовується кома як роздільник дробів, можна вказати decimal=','
    # або виконати очищення після завантаження.
    df = pd.read_csv(file_path, sep=',', decimal='.')
    df.rename(columns=yearly_columns_map, inplace=True)
    return df


def remove_duplicates_and_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Видаляє дублікати та рядки з пропущеними значеннями.
    """
    df = df.drop_duplicates()
    df = df.dropna()
    return df


def remove_outliers(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Видаляє рядки з викидами для заданих числових колонок за методом IQR.
    """
    df_clean = df.copy()
    for col in cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean


def check_stationarity(series: pd.Series, alpha: float = 0.05) -> (bool, float):
    """
    Перевіряє стаціонарність часової серії за допомогою тесту Augmented Dickey-Fuller.
    Якщо серія у вигляді рядків, спершу замінює коми на крапки та конвертує у float.

    :param series: Часова серія (pandas Series).
    :param alpha: Рівень значущості (0.05 за замовчуванням).
    :return: Кортеж (is_stationary, p_value). Якщо серія порожня, повертає (False, np.nan).
    """
    # Якщо серія типу object – замінюємо коми на крапки
    if series.dtype == 'object':
        series = series.astype(str).str.replace(',', '.')
    # Примусово конвертуємо в число
    series = pd.to_numeric(series, errors='coerce')
    s = series.dropna()
    if s.empty:
        print("Попередження: Часова серія порожня. Тест стаціонарності не проводиться.")
        return False, np.nan
    result = adfuller(s)
    p_value = result[1]
    return (p_value < alpha), p_value


def main():
    # 1. Завантаження річних даних
    annual_data_path = "../import_data/annual_data.csv"
    df_annual = load_and_map_annual_data(annual_data_path)

    # Якщо немає колонки 'date', створимо її на основі 'year'
    if 'date' not in df_annual.columns:
        df_annual['date'] = pd.to_datetime(df_annual['year'], format='%Y')

    # 2. Видалення дублікатів та пропусків
    df_clean = remove_duplicates_and_missing(df_annual)

    # 3. Видалення викидів для числових колонок (окрім 'year')
    numeric_cols = [col for col in df_clean.select_dtypes(include=[np.number]).columns if col != 'year']
    df_clean = remove_outliers(df_clean, numeric_cols)

    # 4. Для річних даних сезонне коригування зазвичай не проводять, тому цей крок можна пропустити.
    # Якщо необхідно, додайте окремий модуль для сезонного аналізу річних даних.

    # 5. Перевірка стаціонарності для часової серії. Припустимо, ми використовуємо колонку "real_gdp_change_perc"
    ts = df_clean.copy()
    ts['date'] = pd.to_datetime(ts['date'])
    ts.set_index('date', inplace=True)
    ts = ts[~ts.index.duplicated(keep='first')]
    ts = ts.asfreq('AS')  # AS: початок року

    col_to_test = "real_gdp_change_perc" if "real_gdp_change_perc" in ts.columns else None
    if col_to_test:
        is_stationary, p_val = check_stationarity(ts[col_to_test])
        print(f"Серія '{col_to_test}' є стаціонарною: {is_stationary} (p-value = {p_val:.4f})")
    else:
        print("Обрана колонка для тесту стаціонарності не знайдена.")

    # 6. Збереження оброблених даних у файл
    output_dir = "../output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "processed_annual_data.csv")
    df_clean.to_csv(output_path, index=False)
    print(f"Оброблені річні дані збережено у файл: {output_path}")


if __name__ == "__main__":
    main()
