"""
SberAuto Visit Predictor API
Микросервис для предсказания вероятности целевого действия пользователя на сайте.
Запуск: uvicorn predict_api:app --host 0.0.0.0 --port 8000
"""

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import time
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация FastAPI приложения
app = FastAPI(
    title="SberAuto Visit Predictor",
    description="API для предсказания вероятности заявки на сайте СберАвтоподписка",
    version="1.0.0"
)

# Пути к артефактам модели
MODEL_PATH = "final_model_step3_4.pkl"
PREPROCESSOR_PATH = "preprocessor.pkl"
FEATURE_INFO_PATH = "feature_info.pkl"
THRESHOLD_PATH = "optimal_threshold.pkl"

# Глобальные переменные для кеширования загруженных моделей
model = None
preprocessor = None
feature_info = None
optimal_threshold = None
expected_columns = None
categorical_cols = None


def load_artifacts():
    """Загрузка модели, препроцессора и метаданных при старте приложения."""
    global model, preprocessor, feature_info, optimal_threshold
    global expected_columns, categorical_cols
    
    try:
        logger.info("Загрузка артефактов модели...")
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        feature_info = joblib.load(FEATURE_INFO_PATH)
        optimal_threshold = joblib.load(THRESHOLD_PATH)
        
        expected_columns = feature_info["feature_cols"]
        categorical_cols = feature_info.get("categorical_features", [])
        
        logger.info(f"Модель загружена: {type(model).__name__}")
        logger.info(f"Ожидаемые признаки: {len(expected_columns)} шт.")
        logger.info(f"Порог классификации: {optimal_threshold:.4f}")
        
    except FileNotFoundError as e:
        logger.error(f"Файл артефакта не найден: {e}")
        raise
    except Exception as e:
        logger.error(f"Ошибка при загрузке артефактов: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Загрузка артефактов при запуске приложения."""
    load_artifacts()


class VisitInput(BaseModel):
    """
    Схема входных данных для одного визита.
    Содержит ТОЛЬКО сырые признаки, которые пользователь может предоставить.
    Все производные признаки вычисляются автоматически.
    """
    # === Сырые UTM-параметры ===
    utm_source: str = Field(..., description="Источник трафика, например: yandex, google")
    utm_medium: str = Field(..., description="Тип трафика: organic, cpc, referral и др.")
    utm_campaign: str = Field(..., description="Название рекламной кампании")
    utm_adcontent: str = Field(..., description="Идентификатор рекламного контента")
    utm_keyword: str = Field(..., description="Ключевое слово поиска")
    
    # === Сырые характеристики устройства ===
    device_category: str = Field(..., description="Тип устройства: desktop, mobile, tablet")
    device_os: str = Field(..., description="Операционная система")
    device_brand: str = Field(..., description="Бренд устройства")
    device_browser: str = Field(..., description="Браузер пользователя")
    device_screen_resolution: Optional[str] = Field(
        None, 
        description="Разрешение экрана в формате 'ШхВ', например: '1920x1080'"
    )
    
    # === Сырая геолокация ===
    geo_country: str = Field(..., description="Страна пользователя")
    geo_city: str = Field(..., description="Город пользователя")
    
    # === Сырые временные метки (строки) ===
    visit_date: Optional[str] = Field(
        None, 
        description="Дата визита в формате YYYY-MM-DD (вычисляет час, день недели и т.д.)"
    )
    visit_time: Optional[str] = Field(
        None, 
        description="Время визита в формате HH:MM:SS (вычисляет hour, is_night и т.д.)"
    )
    
    # === Сырые поведенческие признаки (агрегированные из ga_hits) ===
    visit_number: int = Field(..., ge=1, description="Номер визита пользователя")
    hit_count: int = Field(..., ge=0, description="Общее количество событий (хитов) в сессии")
    unique_pages: int = Field(..., ge=0, description="Количество уникальных просмотренных страниц")
    event_count: int = Field(..., ge=0, description="Количество событий с event_action")
    avg_time_between_hits: float = Field(..., ge=0, description="Среднее время между хитами в секундах")
    
    # === Сырые флаги событий (has_event_*) — топ-20 наиболее частых ===
    # Эти флаги формируются при агрегации логов, пользователь/клиент передаёт их как есть
    has_event_view_card: Optional[int] = Field(0, ge=0, le=1)
    has_event_view_new_card: Optional[int] = Field(0, ge=0, le=1)
    has_event_sub_landing: Optional[int] = Field(0, ge=0, le=1)
    has_event_go_to_car_card: Optional[int] = Field(0, ge=0, le=1)
    has_event_sub_view_cars_click: Optional[int] = Field(0, ge=0, le=1)
    # Добавьте остальные has_event_* признаки по необходимости из вашего датасета
    # ...
    
    # === Производные признаки (вычисляются автоматически, но можно переопределить) ===
    is_organic: Optional[int] = Field(None, ge=0, le=1, description="Флаг органического трафика")
    campaign_len: Optional[int] = Field(None, ge=0, description="Длина строки utm_campaign")
    keyword_len: Optional[int] = Field(None, ge=0, description="Длина строки utm_keyword")
    keyword_word_count: Optional[int] = Field(None, ge=0, description="Количество слов в utm_keyword")
    screen_w: Optional[int] = Field(None, ge=0, description="Ширина экрана")
    screen_h: Optional[int] = Field(None, ge=0, description="Высота экрана")
    screen_area: Optional[int] = Field(None, ge=0, description="Площадь экрана")
    screen_ratio: Optional[float] = Field(None, description="Соотношение сторон экрана")
    geo_city_freq: Optional[int] = Field(None, ge=0, description="Частота города в данных")
    hour: Optional[int] = Field(None, ge=0, le=23, description="Час визита")
    day_of_week: Optional[int] = Field(None, ge=0, le=6, description="День недели (0=Пн)")
    month: Optional[int] = Field(None, ge=1, le=12, description="Месяц визита")
    is_weekend: Optional[int] = Field(None, ge=0, le=1, description="Флаг выходного дня")
    is_night: Optional[int] = Field(None, ge=0, le=1, description="Флаг ночного времени (00-06)")
    is_holiday: Optional[int] = Field(None, ge=0, le=1, description="Флаг праздника РФ")


class PredictionOutput(BaseModel):
    """Схема выходных данных предсказания."""
    prediction: int = Field(..., description="Бинарное предсказание: 0 или 1")
    probability: float = Field(..., description="Вероятность положительного класса")
    threshold: float = Field(..., description="Порог классификации")
    latency_ms: float = Field(..., description="Время обработки запроса в миллисекундах")
    timestamp: str = Field(..., description="Временная метка ответа в ISO-формате")
    warning: Optional[str] = Field(None, description="Предупреждение, если есть")


@app.post("/predict", response_model=PredictionOutput)
async def predict_visit(visit: VisitInput):
    """
    Предсказание вероятности целевого действия для одного визита.
    
    Параметры:
        visit: VisitInput — данные визита в формате JSON
        
    Возвращает:
        PredictionOutput — предсказание, вероятность и метаданные
    """
    start_time = time.time()
    
    try:
        # 1. Преобразование входных данных в DataFrame
        visit_dict = visit.dict(exclude_unset=True)
        df_input = pd.DataFrame([visit_dict])
        
        # 2. Вычисление производных признаков, если не переданы явно
        df_input = _compute_derived_features(df_input)
        
        # 3. Выравнивание столбцов с обучающим набором
        df_input = df_input.reindex(columns=expected_columns, fill_value=0)
        
        # 4. Приведение категориальных признаков к строковому типу
        for col in categorical_cols:
            if col in df_input.columns:
                df_input[col] = df_input[col].astype(str)
        
        # 5. Применение препроцессинга (тот же, что при обучении)
        X_processed = preprocessor.transform(df_input)
        
        # 6. Получение вероятности от модели
        proba_array = model.predict_proba(X_processed)
        probability = float(proba_array[0, 1])
        
        # 7. Бинарное предсказание по фиксированному порогу
        prediction = int(probability >= optimal_threshold)
        
        # 8. Вычисление метрик
        latency_ms = (time.time() - start_time) * 1000
        timestamp = datetime.now().isoformat()
        
        # 9. Формирование ответа
        response = PredictionOutput(
            prediction=prediction,
            probability=round(probability, 4),
            threshold=round(optimal_threshold, 4),
            latency_ms=round(latency_ms, 2),
            timestamp=timestamp
        )
        
        # Логирование успешного запроса
        logger.info(
            f"Предсказание: pred={prediction}, prob={probability:.4f}, "
            f"latency={latency_ms:.2f}ms"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Ошибка при предсказании: {str(e)}", exc_info=True)
        latency_ms = (time.time() - start_time) * 1000
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "latency_ms": round(latency_ms, 2),
                "timestamp": datetime.now().isoformat()
            }
        )


def _compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисление производных признаков из сырых данных.
    Применяет ту же логику, что и при обучении модели.
    """
    df = df.copy()
    
    # === Временные признаки ===
    if 'visit_time' in df.columns and df['visit_time'].notna().any():
        try:
            df['visit_time'] = pd.to_datetime(df['visit_time'], errors='coerce')
            df['hour'] = df['visit_time'].dt.hour.fillna(12).astype(int)
            df['is_night'] = (df['hour'] < 6).astype(int)
        except Exception:
            df['hour'] = df.get('hour', 12)
            df['is_night'] = df.get('is_night', 0)
    
    if 'visit_date' in df.columns and df['visit_date'].notna().any():
        try:
            df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce')
            df['day_of_week'] = df['visit_date'].dt.dayofweek.fillna(0).astype(int)
            df['month'] = df['visit_date'].dt.month.fillna(1).astype(int)
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Праздники РФ 2021 (пример)
            holidays = [
                '2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04',
                '2021-01-05', '2021-01-06', '2021-01-07', '2021-01-08',
                '2021-02-23', '2021-03-08', '2021-05-01', '2021-05-09',
                '2021-05-10', '2021-06-12', '2021-06-14', '2021-11-04'
            ]
            df['date_str'] = df['visit_date'].dt.strftime('%Y-%m-%d')
            df['is_holiday'] = df['date_str'].isin(holidays).astype(int)
            df.drop(columns=['date_str'], inplace=True, errors='ignore')
        except Exception:
            df['day_of_week'] = df.get('day_of_week', 0)
            df['month'] = df.get('month', 1)
            df['is_weekend'] = df.get('is_weekend', 0)
            df['is_holiday'] = df.get('is_holiday', 0)
    
    # === UTM-признаки ===
    if 'utm_medium' in df.columns:
        organic_mediums = ['organic', '(none)', 'referral']
        df['is_organic'] = df['utm_medium'].fillna('').isin(organic_mediums).astype(int)
    
    if 'utm_campaign' in df.columns:
        df['campaign_len'] = df['utm_campaign'].fillna('').str.len()
    
    if 'utm_keyword' in df.columns:
        df['keyword_len'] = df['utm_keyword'].fillna('').str.len()
        df['keyword_word_count'] = df['utm_keyword'].fillna('').str.split().str.len()
    
    # === Признаки устройства ===
    if 'device_screen_resolution' in df.columns:
        def _parse_resolution(res):
            if pd.isna(res) or not isinstance(res, str) or 'x' not in res:
                return 414, 896
            try:
                w, h = res.split('x')
                return int(w), int(h)
            except (ValueError, IndexError):
                return 414, 896
        
        resolutions = df['device_screen_resolution'].apply(_parse_resolution)
        df['screen_w'] = resolutions.apply(lambda x: x[0])
        df['screen_h'] = resolutions.apply(lambda x: x[1])
        df['screen_area'] = df['screen_w'] * df['screen_h']
        df['screen_ratio'] = df.apply(
            lambda row: row['screen_w'] / row['screen_h'] 
            if row['screen_h'] > 0 else 0, 
            axis=1
        )
    
    # === Географические признаки ===
    if 'geo_city' in df.columns:
        # Частота города — упрощённая версия (в продакшене лучше использовать precomputed lookup)
        city_counts = df['geo_city'].value_counts()
        df['geo_city_freq'] = df['geo_city'].map(city_counts).fillna(1)
    
    # === Заполнение пропусков для числовых признаков ===
    numeric_defaults = {
        'hit_count': 1, 'unique_pages': 1, 'event_count': 0,
        'avg_time_between_hits': 0, 'visit_number': 1,
        'screen_w': 414, 'screen_h': 896, 'screen_area': 360000,
        'screen_ratio': 0.46, 'geo_city_freq': 1,
        'campaign_len': 0, 'keyword_len': 0, 'keyword_word_count': 0,
        'hour': 12, 'day_of_week': 0, 'month': 1,
        'is_weekend': 0, 'is_night': 0, 'is_holiday': 0, 'is_organic': 0
    }
    
    for col, default in numeric_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)
    
    # === Бинарные флаги событий ===
    event_cols = [col for col in df.columns if col.startswith('has_event_')]
    for col in event_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    
    return df


@app.get("/health")
async def health_check():
    """Эндпоинт проверки работоспособности сервиса."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/")
async def root():
    """Корневой эндпоинт с информацией об API."""
    return {
        "service": "SberAuto Visit Predictor",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict (POST)"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "predict_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )