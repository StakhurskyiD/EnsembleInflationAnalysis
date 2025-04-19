#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py
Основна точка входу в проєкт 'EnsembleInflationAnalysis'.
Покроково:
1. Завантаження та попередня обробка макроекономічних даних
2. Побудова/навчання ансамблевих моделей (RandomForest, XGBoost)
3. Оцінка та порівняння моделей
4. Вивід результатів (важливість ознак, помилка прогнозу, тощо)
"""

import os
import sys
import logging
import numpy as np
import pandas as pd

# Можливі власні модулі
# from data_preprocessing import load_and_clean_data
# from ensemble_model import train_random_forest, train_xgboost, evaluate_model

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def main():
    logging.info("===== START: EnsembleInflationAnalysis =====")

    # 1. Завантаження даних
    data_path = os.path.join('data', 'macro_data.csv')
    if not os.path.exists(data_path):
        logging.error(f"Файл {data_path} не знайдено! Переконайтеся, що він існує.")
        sys.exit(1)
    df = pd.read_csv(data_path, parse_dates=['date'])

    logging.info(f"Дані успішно завантажено. Розмір: {df.shape}")

    # 2. Попередня обробка (демонстраційний приклад):
    #    - Сортування за датою
    #    - Видалення пропусків (або інший спосіб обробки)
    df.sort_values('date', inplace=True)
    df.dropna(inplace=True)

    # Припустимо, що інфляцію (inflation) ми прогнозуємо на основі
    # певних макроекономічних чинників: gdp, m2, unemployment, policy_rate, ex_rate
    feature_cols = ['gdp', 'm2', 'unemployment', 'policy_rate', 'ex_rate']
    target_col = 'inflation'

    # Перевіримо, чи всі колонки є в DataFrame
    for col in feature_cols + [target_col]:
        if col not in df.columns:
            logging.warning(f"Колонка '{col}' відсутня у DataFrame, перевірте структуру macro_data.csv")

    # 3. Формування матриці ознак та вектора цілі
    X = df[feature_cols].values
    y = df[target_col].values

    # 4. Розбиваємо дані на тренувальну та тестову вибірки
    #    (для реальної задачі можна використати TimeSeriesSplit)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    logging.info(f"Тренувальна вибірка: {X_train.shape}, Тестова вибірка: {X_test.shape}")

    # 5. Навчання Random Forest (попереднє, без тонкого налаштування)
    logging.info("=== Навчання RandomForestRegressor ===")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    logging.info(f"RandomForest -> MSE: {mse_rf:.3f}, R2: {r2_rf:.3f}")

    # 6. Навчання XGBoost
    logging.info("=== Навчання XGBoost ===")
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb.fit(X_train, y_train)

    y_pred_xgb = xgb.predict(X_test)
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    logging.info(f"XGBoost -> MSE: {mse_xgb:.3f}, R2: {r2_xgb:.3f}")

    # 7. Порівняльний висновок
    logging.info("\n--- Порівняння моделей ---")
    logging.info(f"RandomForest:\tMSE={mse_rf:.3f}\tR2={r2_rf:.3f}")
    logging.info(f"XGBoost:\tMSE={mse_xgb:.3f}\tR2={r2_xgb:.3f}")

    # 8. Аналіз важливості ознак (Feature Importances) для RandomForest
    importances_rf = rf.feature_importances_
    sorted_idx_rf = np.argsort(importances_rf)[::-1]

    logging.info("--- Важливість чинників (RandomForest) ---")
    for idx in sorted_idx_rf:
        feature_name = feature_cols[idx]
        importance_val = importances_rf[idx]
        logging.info(f"{feature_name}: {importance_val:.4f}")

    # (Опційно) те саме для XGBoost
    importances_xgb = xgb.feature_importances_
    sorted_idx_xgb = np.argsort(importances_xgb)[::-1]
    logging.info("--- Важливість чинників (XGBoost) ---")
    for idx in sorted_idx_xgb:
        feature_name = feature_cols[idx]
        importance_val = importances_xgb[idx]
        logging.info(f"{feature_name}: {importance_val:.4f}")

    logging.info("===== END: EnsembleInflationAnalysis =====")


if __name__ == "__main__":
    main()
