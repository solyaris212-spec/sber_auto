#!/usr/bin/env python3
"""
Скрипт для предсказания целевого действия пользователя на сайте.
Принимает JSON с атрибутами визита, возвращает JSON с предсказанием (0 или 1).

Использование:
    python predict.py --input visit_data.json --output prediction.json
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import json
import sys
import os
from datetime import datetime


class SiteVisitPredictor:
    """Загрузка модели и препроцессора, выполнение предсказаний."""

    def __init__(self, model_path, preprocessor_path, feature_info_path, threshold):
        # Проверка существования файлов перед загрузкой
        for path, name in [(model_path, "модели"), (preprocessor_path, "препроцессора"), (feature_info_path, "информации о признаках")]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Файл {name} не найден: {path}")

        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.feature_info = joblib.load(feature_info_path)
        self.threshold = threshold
        self.expected_columns = self.feature_info["feature_cols"]
        self.categorical_cols = self.feature_info.get("categorical_features", [])

    def predict(self, visit_data: dict) -> dict:
        # Преобразование входного словаря в однострочный DataFrame
        df_visit = pd.DataFrame([visit_data])

        # Выравнивание столбцов с обучающим набором. Отсутствующие признаки заполняются 0.
        df_visit = df_visit.reindex(columns=self.expected_columns, fill_value=0)

        # Приведение категориальных признаков к строковому типу для корректной обработки OneHotEncoder
        for col in self.categorical_cols:
            if col in df_visit.columns:
                df_visit[col] = df_visit[col].astype(str)

        # Применение того же препроцессинга, что использовался при обучении
        X_processed = self.preprocessor.transform(df_visit)

        # Получение вероятности положительного класса
        probability = float(self.model.predict_proba(X_processed)[0, 1])
        prediction = 1 if probability >= self.threshold else 0

        return {
            "prediction": int(prediction),
            "probability": round(probability, 4),
            "threshold": round(self.threshold, 4),
            "timestamp": datetime.now().isoformat()
        }


def main():
    parser = argparse.ArgumentParser(description="Предсказание целевого действия на сайте")
    parser.add_argument("--input", type=str, required=True, help="Путь к входному JSON файлу")
    parser.add_argument("--output", type=str, default="prediction.json", help="Путь к выходному JSON файлу")
    parser.add_argument("--model", type=str, default="final_model_step3_4.pkl", help="Путь к файлу модели")
    parser.add_argument("--preprocessor", type=str, default="preprocessor.pkl", help="Путь к файлу препроцессора")
    parser.add_argument("--feature-info", type=str, default="feature_info.pkl", help="Путь к файлу с информацией о признаках")
    parser.add_argument("--threshold", type=float, default=0.127, help="Порог классификации")

    args = parser.parse_args()

    try:
        predictor = SiteVisitPredictor(
            model_path=args.model,
            preprocessor_path=args.preprocessor,
            feature_info_path=args.feature_info,
            threshold=args.threshold
        )
    except Exception as e:
        print(f"Ошибка инициализации: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            visit_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Ошибка парсинга JSON: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"Входной файл не найден: {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        result = predictor.predict(visit_data)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Prediction: {result['prediction']}")
        print(f"Probability: {result['probability']}")
    except Exception as e:
        print(f"Ошибка предсказания: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
