# Instagram Image Analyzer

Сервис для анализа качества изображений для Instagram. Помогает оценить изображения по ключевым параметрам:
- Размытие
- Яркость
- Контраст
- Шум

## Особенности сервиса

- Анализ качества изображений (размытие, яркость, контраст, шум)
- Автоматическая классификация типов изображений
- Снисходительные алгоритмы анализа для креативного контента
- Учет художественного размытия фона при оценке размытия
- Анализ изображений по регионам для лучшей оценки
- Интеграция с make.com для автоматизации рабочих процессов

### Новые возможности

- **Контекстный анализ изображений** с использованием CLIP (Contrastive Language-Image Pretraining)
- **Определение творческого процесса** в изображениях
- **Выявление коммерческого контента** для фильтрации
- **Оценка соответствия профилю** творческого контента
- **Расширенный анализ контекста** для более точной фильтрации

## Использование API

### Анализ изображений

```bash
curl -X POST "https://instagram-image-analyzer.onrender.com/analyze-content" \
  -H "Content-Type: application/json" \
  -d '{"image_urls": ["https://example.com/image1.jpg", "https://example.com/image2.jpg"]}'
```

### Пример ответа

```json
{
  "image_scores": [
    {
      "url": "https://example.com/image1.jpg",
      "quality_score": 4.5,
      "blur_score": 4.2,
      "brightness_score": 4.7,
      "contrast_score": 4.1,
      "noise_score": 4.8,
      "image_type": "портрет",
      "detailed_metrics": {
        "resolution": [1080, 1080],
        "aspect_ratio": 1.0,
        "color_mode": "RGB",
        "blur_details": {"laplacian_variance": 21.0},
        "brightness_details": {"mean_brightness": 132.5},
        "contrast_details": {"std_dev": 67.2},
        "image_type": "портрет"
      }
    }
  ],
  "average_quality": 4.5
}
```

## Локальный запуск

1. Клонируйте репозиторий
2. Установите зависимости: `pip install -r requirements.txt`
3. Запустите сервер: `uvicorn main:app --reload`
4. Откройте http://localhost:8000/docs для доступа к Swagger UI

## Деплой на Render

1. Создайте новый веб-сервис на [Render](https://render.com)
2. Подключите репозиторий
3. Выберите Python 3.11 как runtime
4. Используйте `pip install -r requirements.txt` как build command
5. Используйте `uvicorn main:app --host 0.0.0.0 --port $PORT` как start command

## Интеграция с make.com

Сервис поддерживает интеграцию с make.com для автоматизации процесса анализа изображений из Instagram.

### Параметры API для интеграции

При вызове эндпоинта `/analyze` можно передать следующие параметры:

- `image_url`: URL изображения для анализа
- `image_file`: Файл изображения (для прямой загрузки)
- `creator_data`: Дополнительные данные о креаторе и метрики из Instagram (опционально)

### Пример ответа API:

```json
{
  "blur_score": 4.25,
  "brightness_score": 3.85,
  "contrast_score": 4.10,
  "noise_score": 4.50,
  "overall_score": 4.18,
  "image_type": "portrait",
  "type_confidence": 0.87,
  "status": "QualityChecked",
  "context": {
    "subject_человек": 0.75,
    "activity_творческий процесс": 0.62,
    "setting_студия": 0.58,
    ...
  },
  "timestamp": "2023-06-15T14:30:45.123456"
}
```

### Статусы качества изображения:

- `QualityChecked`: Изображение прошло проверку качества (overall_score >= 3.0)
- `Rejected`: Изображение не прошло проверку качества (overall_score < 3.0)

### Настройка сценария в make.com:

1. Соберите данные из Instagram через модули Instagram Business
2. Передайте URL изображения и данные о креаторе в наш API
3. Получите результаты анализа и статус
4. Обновите статус в первичной таблице и/или добавьте данные во вторичную таблицу

## API Endpoints

### 1. Анализ качества изображения

**POST /analyze**

Принимает изображение и анализирует его качество, а также извлекает контекстную информацию.

Параметры:
- `image_url`: URL изображения
- `image_file`: Файл изображения
- `creator_data`: (опционально) Данные о креаторе

Ответ:
```json
{
  "blur_score": 4.25,
  "brightness_score": 3.85,
  "contrast_score": 4.10,
  "noise_score": 4.50,
  "overall_score": 4.18,
  "image_type": "portrait",
  "type_confidence": 0.87,
  "status": "QualityChecked",
  "context": {
    "subject_человек": 0.75,
    "activity_творческий процесс": 0.62,
    "setting_студия": 0.58,
    ...
  },
  "timestamp": "2023-06-15T14:30:45.123456"
}
```

### 2. Извлечение контекста изображения

**POST /analyze-context**

Извлекает контекстную информацию из изображения без анализа качества.

Параметры:
- `image_url`: URL изображения
- `image_file`: Файл изображения

Ответ:
```json
{
  "context": {
    "subject_человек": 0.75,
    "activity_творческий процесс": 0.62,
    ...
  },
  "image_type": "creative_process",
  "type_confidence": 0.87,
  "top_categories": {
    "subject": [
      ["человек", 0.75],
      ["объект", 0.15]
    ],
    "activity": [
      ["творческий процесс", 0.62],
      ["работа", 0.25]
    ],
    ...
  },
  "timestamp": "2023-06-15T14:30:45.123456"
}
```

### Статусы качества изображения:

- `QualityChecked`: Изображение прошло проверку качества (overall_score >= 3.0)
- `Rejected`: Изображение не прошло проверку качества (overall_score < 3.0)

### Настройка сценария в make.com:

1. Соберите данные из Instagram через модули Instagram Business
2. Передайте URL изображения и данные о креаторе в наш API
3. Получите результаты анализа и статус
4. Обновите статус в первичной таблице и/или добавьте данные во вторичную таблицу 