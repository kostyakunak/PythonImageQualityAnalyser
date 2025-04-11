# Instagram Image Analyzer

Сервис для анализа качества изображений для Instagram. Помогает оценить изображения по ключевым параметрам:
- Размытие
- Яркость
- Контраст
- Шум

## Особенности

- Интеллектуальный анализ размытия с учетом художественных эффектов (боке)
- Снисходительный алгоритм оценки для поддержки независимых креаторов
- Классификация изображений по типам
- Простой и понятный API

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