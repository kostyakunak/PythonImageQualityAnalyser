from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from typing import List, Dict, Any
import cv2
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Instagram Content Analyzer")

# Типы изображений, которые мы распознаем
IMAGE_TYPES = {
    "portrait": "портрет",
    "landscape": "пейзаж",
    "abstract": "абстрактное",
    "product": "продукт",
    "other": "другое"
}

# Веса параметров качества для разных типов изображений
QUALITY_WEIGHTS = {
    "portrait": {
        "blur": 0.40,      # Снижена критичность размытия для портретов
        "brightness": 0.20, # Увеличена важность яркости
        "contrast": 0.20,   # Увеличена важность контраста
        "noise": 0.20       # Допускаем больше шума
    },
    "landscape": {
        "blur": 0.35,       # Размытие менее критично для пейзажей
        "brightness": 0.25, # Яркость важна для пейзажей
        "contrast": 0.25,   # Контраст важен для пейзажей
        "noise": 0.15       # Шум не так заметен в пейзажах
    },
    "abstract": {
        "blur": 0.25,      # Размытие почти неважно для абстрактного искусства
        "brightness": 0.30, # Яркость и цвета важны
        "contrast": 0.30,   # Контраст очень важен
        "noise": 0.15       # Шум может быть частью художественного замысла
    },
    "product": {
        "blur": 0.40,       # Чёткость важна, но не критична
        "brightness": 0.25, # Яркость важна для товаров
        "contrast": 0.25,   # Контраст важен для товаров
        "noise": 0.10       # Шум обычно не проблема
    },
    "other": {
        "blur": 0.35,       # Усредненная важность размытия
        "brightness": 0.25, # Увеличена важность яркости
        "contrast": 0.25,   # Увеличена важность контраста
        "noise": 0.15       # Средняя важность шума
    }
}

class ImageAnalysisRequest(BaseModel):
    image_urls: List[str]

class ImageScore(BaseModel):
    url: str
    quality_score: float
    blur_score: float
    brightness_score: float
    contrast_score: float
    noise_score: float
    image_type: str = "other"
    detailed_metrics: Dict[str, Any]

class AnalysisResponse(BaseModel):
    image_scores: List[ImageScore]
    average_quality: float

@app.get("/")
async def root():
    return {"status": "active", "service": "Instagram Content Analyzer"}

@app.post("/analyze-content", response_model=AnalysisResponse)
async def analyze_content(request: ImageAnalysisRequest):
    try:
        logger.info(f"Получен запрос на анализ {len(request.image_urls)} изображений")
        
        image_scores = []
        for url in request.image_urls:
            try:
                response = requests.get(url, timeout=10)
                img = Image.open(BytesIO(response.content))
                score_result = analyze_image_quality(img, url)
                image_scores.append(score_result)
            except Exception as e:
                logger.error(f"Ошибка при анализе изображения {url}: {str(e)}")
                image_scores.append(ImageScore(
                    url=url,
                    quality_score=1.0,
                    blur_score=1.0,
                    brightness_score=1.0,
                    contrast_score=1.0,
                    noise_score=1.0,
                    image_type="other",
                    detailed_metrics={"error": str(e)}
                ))
        
        average_quality = np.mean([score.quality_score for score in image_scores])
        
        return AnalysisResponse(
            image_scores=image_scores,
            average_quality=float(average_quality)
        )
    
    except Exception as e:
        logger.error(f"Общая ошибка обработки: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def simple_classify_image(img, gray_img):
    """Простая классификация изображения на основе базовых характеристик"""
    # Получаем размеры изображения
    width, height = img.size
    aspect_ratio = width / height
    
    # Преобразуем изображение в массив для анализа цветов
    img_array = np.array(img)
    
    # Вычисляем общие метрики изображения
    global_contrast = np.std(gray_img) / 128
    
    # Находим границы
    edges = cv2.Canny(gray_img, 100, 200)
    edge_density = np.count_nonzero(edges) / (gray_img.shape[0] * gray_img.shape[1])
    
    # Анализ цветового разнообразия (полезно для определения абстрактных изображений)
    color_std = np.std(img_array, axis=(0, 1)) / 255 if len(img_array.shape) == 3 else np.array([0])
    color_variety = np.mean(color_std)
    
    # Анализ цветовой насыщенности (для абстрактных изображений)
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        # Преобразуем RGB в HSV для анализа насыщенности
        try:
            hsv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            saturation = np.mean(hsv_img[:, :, 1]) / 255
        except:
            saturation = 0
    else:
        saturation = 0
    
    # Проверяем наличие лиц (если есть лица, вероятно это портрет)
    has_face = False
    try:
        # Используем простейший классификатор лиц Haar
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
        
        if len(faces) > 0:
            face_area = sum(w * h for (_, _, w, h) in faces)
            image_area = gray_img.shape[0] * gray_img.shape[1]
            
            # Если лицо занимает значительную часть изображения (>3%)
            if face_area / image_area > 0.03:
                has_face = True
                return "portrait"
    except:
        pass
    
    # Проверка на животных (например, леопард) - используем эвристики
    # Животные часто имеют текстуру (например, пятна, полосы)
    texture_variation = edge_density * global_contrast
    
    # Более точный анализ для леопардов и похожих животных
    # Ищем пятнистую текстуру, используя более сложную обработку
    has_animal_texture = False
    try:
        # Бинаризация и анализ связных компонентов
        blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)
        
        # Находим контуры (пятна)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Фильтруем контуры по размеру (пятна обычно маленькие/средние)
        small_contours = [c for c in contours if 100 < cv2.contourArea(c) < 5000]
        
        # Если много небольших контуров и текстура неоднородна - вероятно, это животное с пятнами
        if len(small_contours) > 10 and texture_variation > 0.05:
            has_animal_texture = True
    except:
        pass
    
    # Если высокая текстура и изображение не широкое, и есть признаки животного
    if (texture_variation > 0.05 and 0.6 < aspect_ratio < 1.7 and 
        (has_animal_texture or (np.count_nonzero(cv2.adaptiveThreshold(gray_img, 255, 
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY, 11, 2)) > gray_img.size * 0.1))):
        return "portrait"  # Классифицируем как портрет (животного)
        
    # Анализ абстрактных изображений
    # Проверяем разнообразие и распределение цветов
    is_abstract = False
    
    # Абстрактные изображения часто имеют:
    # 1. Высокую насыщенность и яркие цвета
    # 2. Необычное распределение цветов
    # 3. Отсутствие реалистичных элементов
    if ((color_variety > 0.2 and saturation > 0.4) or 
        (global_contrast > 0.45 and edge_density > 0.1 and not has_face and not has_animal_texture) or
        (saturation > 0.5 and color_variety > 0.15)):
        is_abstract = True
        
    # Дополнительная проверка для абстрактных изображений с красочными потоками
    # (как в примере с красно-синими разводами)
    if len(img_array.shape) == 3:
        try:
            # Анализируем цветовые каналы
            channel_means = np.mean(img_array, axis=(0, 1))
            channel_stds = np.std(img_array, axis=(0, 1))
            
            # Яркие абстрактные изображения часто имеют существенные различия между каналами
            channel_diff = np.max(channel_means) - np.min(channel_means)
            
            # Если есть значительная разница между каналами и общая высокая насыщенность
            if channel_diff > 30 and saturation > 0.35 and color_variety > 0.15:
                is_abstract = True
        except:
            pass
    
    if is_abstract:
        return "abstract"
    
    # Признаки пейзажа:
    # - Широкоформатное изображение (широкое)
    # - Обычно имеет горизонтальные линии
    if aspect_ratio > 1.4 and global_contrast < 0.4:
        return "landscape"
    
    # Признаки продукта:
    # - Чёткие границы
    # - Средний формат
    if edge_density > 0.08 and 0.8 < aspect_ratio < 1.3:
        return "product"
        
    # По умолчанию
    return "other"

def analyze_image_quality(img, url):
    """Анализирует качество изображения по базовым параметрам"""
    
    # Конвертируем в numpy array
    img_array = np.array(img)
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    
    # Конвертируем в grayscale для анализа
    if len(img_array.shape) == 3:
        gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray_img = img_array
    
    # Определяем тип изображения с помощью простой классификации
    image_type = simple_classify_image(img, gray_img)
    logger.info(f"Изображение {url} классифицировано как: {IMAGE_TYPES[image_type]}")
    
    # Анализ базовых параметров
    blur_score = analyze_blur(gray_img, image_type)
    brightness_score = analyze_brightness(img_array)
    contrast_score = analyze_contrast(gray_img)
    noise_score = analyze_noise(gray_img)
    
    # Подробные метрики
    detailed_metrics = {
        "resolution": img.size,
        "aspect_ratio": img.size[0] / img.size[1],
        "color_mode": img.mode,
        "blur_details": {"laplacian_variance": blur_score * 5},
        "brightness_details": {"mean_brightness": brightness_score * 255},
        "contrast_details": {"std_dev": contrast_score * 128},
        "image_type": IMAGE_TYPES[image_type]
    }
    
    # Применяем веса в зависимости от типа изображения
    weights = QUALITY_WEIGHTS[image_type]
    
    quality_score = (
        blur_score * weights["blur"] +
        brightness_score * weights["brightness"] +
        contrast_score * weights["contrast"] +
        noise_score * weights["noise"]
    ) * 5
    
    quality_score = max(1.0, min(5.0, quality_score))
    
    return ImageScore(
        url=url,
        quality_score=float(quality_score),
        blur_score=float(blur_score * 5),
        brightness_score=float(brightness_score * 5),
        contrast_score=float(contrast_score * 5),
        noise_score=float(noise_score * 5),
        image_type=IMAGE_TYPES[image_type],
        detailed_metrics=detailed_metrics
    )

def analyze_blur(gray_img, image_type="other"):
    """Анализирует размытие изображения с фокусом на наличие минимального количества четких линий"""
    height, width = gray_img.shape
    
    # Определяем минимальную площадь четких линий (в процентах от общей площади)
    # Если хотя бы 5% изображения содержит четкие линии - считаем это достаточным
    MIN_EDGES_PERCENT = 0.05
    
    # Находим края на изображении (четкие линии)
    edges = cv2.Canny(gray_img, 100, 200)
    edge_pixels = np.count_nonzero(edges)
    total_pixels = gray_img.size
    edge_percent = edge_pixels / total_pixels
    
    # Вычисляем общую четкость изображения (вариация Лапласиана)
    global_laplacian_var = cv2.Laplacian(gray_img, cv2.CV_64F).var()
    
    # Находим участки с высокой детализацией
    # Разделим изображение на сетку 3x3
    h_step = height // 3
    w_step = width // 3
    
    # Для каждого региона оцениваем четкость
    region_sharpness = []
    max_sharpness = 0
    
    for i in range(3):
        for j in range(3):
            # Получаем регион
            region = gray_img[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
            
            # Вычисляем вариацию Лапласиана для региона (мера четкости)
            laplacian_var = cv2.Laplacian(region, cv2.CV_64F).var()
            
            # Находим края в регионе
            region_edges = cv2.Canny(region, 100, 200)
            edge_density = np.count_nonzero(region_edges) / region.size
            
            # Комбинированная метрика четкости региона
            sharpness = (laplacian_var / 1000) * (1 + edge_density * 10)
            region_sharpness.append(sharpness)
            max_sharpness = max(max_sharpness, sharpness)
    
    # Находим количество "четких" регионов (с хорошей детализацией)
    sharp_regions_count = sum(1 for s in region_sharpness if s > 0.3)
    
    # Базовая оценка размытия
    # Если на изображении достаточно четких линий или есть хотя бы один очень четкий регион
    if edge_percent > MIN_EDGES_PERCENT or max_sharpness > 0.8 or sharp_regions_count > 0:
        # Изображение имеет достаточно четких деталей
        # Оценка будет зависеть от максимальной четкости региона
        blur_score = min(1.0, max(0.8, max_sharpness))
    else:
        # Недостаточно четких линий
        blur_score = max(0.2, edge_percent * 10)  # Пропорционально количеству краев
    
    # Дополнительные проверки для различных типов изображений
    
    # Для портретов приемлем эффект боке (размытый фон, четкий объект)
    if image_type == "portrait" and (max_sharpness > 0.4 or sharp_regions_count > 0):
        # Если есть хотя бы один четкий регион - вероятно это объект в фокусе
        blur_score = max(0.9, blur_score)
    
    # Для абстрактных изображений размытие часто является художественным приемом
    if image_type == "abstract":
        # Даже при небольшом количестве четких линий считаем это художественным приемом
        blur_score = max(0.9, blur_score)
    
    # Для пейзажей часто важна общая четкость
    if image_type == "landscape" and global_laplacian_var > 300:
        blur_score = max(0.85, blur_score)
    
    return blur_score

def analyze_brightness(img_array):
    """Анализирует яркость изображения"""
    mean_brightness = np.mean(img_array) / 255
    
    # Новая формула с меньшим штрафом за отклонение и минимальной оценкой 0.7
    deviation = abs(mean_brightness - 0.5)
    
    # Применяем более мягкую квадратичную функцию вместо линейной
    # Это даст меньший штраф за небольшие отклонения
    brightness_score = 1.0 - (deviation ** 1.5)
    
    # Дополнительно масштабируем, чтобы минимальная оценка была не ниже 0.7
    brightness_score = 0.7 + 0.3 * brightness_score
    
    return brightness_score

def analyze_contrast(gray_img):
    """Анализирует контраст изображения"""
    std_dev = np.std(gray_img) / 128
    
    # Применяем нелинейную функцию для повышения оценки низкоконтрастных изображений
    # Это обеспечит, что даже при низком стандартном отклонении оценка будет выше
    adjusted_score = 0.6 + 0.4 * (std_dev / (std_dev + 0.3))
    
    # Гарантируем минимальную оценку контраста
    return max(0.6, adjusted_score)

def analyze_noise(gray_img):
    """Анализирует шум на изображении"""
    denoised = cv2.medianBlur(gray_img, 5)
    noise_diff = np.abs(gray_img.astype(np.float32) - denoised.astype(np.float32))
    noise_amount = np.mean(noise_diff) / 255
    noise_score = 1.0 - min(1.0, noise_amount * 10)
    return noise_score

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 