import requests
import json
import sys

# URL сервиса (замените на ваш URL после деплоя)
SERVICE_URL = "http://localhost:8000"  # В локальной разработке
# SERVICE_URL = "https://instagram-analyzer.onrender.com"  # После деплоя

# Тестовый запрос
test_data = {
    "image_urls": [
        "https://pyxis.nymag.com/v1/imgs/2dd/102/30bc3443ce72015af6f79fc0c5436be3b1-DSC08380.rhorizontal.w1100.jpg",  # Фото 1
        "https://www.northcentralcollege.edu/sites/default/files/styles/full_image_large/public/2023-07/studio-art-3.jpg?h=a1e1a043&itok=FhGrz2sS",  # Фото 2
        "https://thumbs.dreamstime.com/b/grey-white-bright-orange-knitted-sweaters-pile-stack-cozy-handmade-clothes-different-knitting-patterns-concept-grey-349958675.jpg"  # Фото 3
    ]
}

def test_url(url):
    """Проверяет, доступно ли изображение по URL"""
    try:
        response = requests.head(url, timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"Ошибка при проверке URL {url}: {str(e)}")
        return False

def test_api():
    print("Тестирование API анализа качества изображений...")
    
    # Предварительная проверка URLs
    print("\nПроверка доступности изображений:")
    for i, url in enumerate(test_data["image_urls"]):
        if test_url(url):
            print(f"✅ Изображение {i+1}: URL доступен")
        else:
            print(f"❌ Изображение {i+1}: URL недоступен - {url}")
    
    # Проверка состояния сервиса
    try:
        response = requests.get(f"{SERVICE_URL}/")
        if response.status_code == 200:
            print(f"\n✅ Сервис активен: {response.json()}")
        else:
            print(f"\n❌ Ошибка проверки состояния: {response.status_code}")
            return
    except Exception as e:
        print(f"\n❌ Ошибка подключения: {str(e)}")
        return
    
    # Тестирование анализа контента
    try:
        response = requests.post(
            f"{SERVICE_URL}/analyze-content",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Анализ успешно выполнен!")
            print(f"\nСредняя оценка качества: {result['average_quality']:.2f}/5.0")
            
            print("\nОценки изображений:")
            for i, img_score in enumerate(result['image_scores']):
                print(f"\nИзображение {i+1}:")
                print(f"  URL: {img_score['url'][-50:]}")  # Показываем только последние 50 символов URL
                print(f"  Тип изображения: {img_score.get('image_type', 'не определен')}")
                print(f"  Общая оценка: {img_score['quality_score']:.2f}/5.0")
                print(f"  Размытие: {img_score['blur_score']:.2f}/5.0")
                print(f"  Яркость: {img_score['brightness_score']:.2f}/5.0")
                print(f"  Контраст: {img_score['contrast_score']:.2f}/5.0")
                print(f"  Шум: {img_score['noise_score']:.2f}/5.0")
                
                print("\n  Детальные метрики:")
                for metric_name, metric_value in img_score['detailed_metrics'].items():
                    print(f"    {metric_name}: {metric_value}")
            
            # Сохраняем полный ответ в JSON-файл для детального анализа
            with open("test_response.json", "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print("\nПолный ответ сохранен в файл test_response.json")
            
        else:
            print(f"❌ Ошибка анализа: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"❌ Ошибка запроса: {str(e)}")

if __name__ == "__main__":
    test_api() 