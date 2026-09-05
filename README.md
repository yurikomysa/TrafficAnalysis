# 🚗 Traffic Analysis System

Система для аналізу автомобільного трафіку в реальному часі з використанням комп'ютерного зору та штучного інтелекту.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.6-green.svg)](https://fastapi.tiangolo.com/)
[![YOLO](https://img.shields.io/badge/YOLO-8.2.0-red.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Зміст

- [Опис](#-опис)
- [Функціонал](#-функціонал)
- [Технології](#-технології)
- [Вимоги](#-вимоги)
- [Встановлення](#-встановлення)
- [Запуск](#-запуск)
- [Використання](#-використання)
- [API Endpoints](#-api-endpoints)
- [Архітектура](#-архітектура)
- [Схема роботи](#-схема-роботи)
- [Діаграма послідовності](#-діаграма-послідовності)
- [Діаграма класів](#-діаграма-класів)
- [Оптимізація](#-оптимізація)
- [Тестування](#-тестування)
- [Можливі проблеми](#-можливі-проблеми)
- [Плани на майбутнє](#-плани-на-майбутнє)
- [Автор](#-автор)
- [Ліцензія](#-ліцензія)

## 🎯 Опис

**Traffic Analysis System** - це високопродуктивна система аналізу дорожнього руху, яка працює в реальному часі. Система здатна виявляти транспортні засоби, відстежувати їх рух, розпізнавати номерні знаки та визначати колір автомобілів.

### Ключові можливості:

- 🎥 Обробка відео з RTSP камер та локальних файлів
- 🚗 Виявлення та трекінг транспортних засобів
- 📋 Розпізнавання номерних знаків
- 🎨 Визначення кольору автомобіля
- 🌐 Веб-інтерфейс для моніторингу
- ⚡ Оптимізація для edge-пристроїв

## ✨ Функціонал

### Основні функції:

- ✅ **Виявлення автомобілів** - використання YOLOv8 для детекції
- ✅ **Унікальні ID** - присвоєння кожному автомобілю унікального ідентифікатора
- ✅ **Номерні знаки** - автоматичне виявлення та розпізнавання номерів
- ✅ **Колір автомобіля** - визначення кольору в HSV просторі
- ✅ **RTSP підтримка** - робота з будь-якими IP камерами
- ✅ **Веб-інтерфейс** - зручний моніторинг через браузер
- ✅ **REST API** - інтеграція з іншими системами
- ✅ **Статистика** - збір та відображення даних про трафік

### Додаткові можливості:

- 🔄 Автоматичне перепідключення при обриві потоку
- 📊 Відображення FPS та кількості автомобілів
- 💾 Збереження статистики
- 🎯 Налаштування порогу впевненості
- 📹 Підтримка різних форматів відео

## 🛠 Технології

| Компонент                        | Технологія                    | Версія |
| ----------------------------------------- | --------------------------------------- | ------------ |
| Детекція авто                 | YOLOv8                                  | 8.2.0        |
| Трекінг                            | Власна реалізація (IoU) | -            |
| Розпізнавання номерів | EasyOCR                                 | 1.7.2        |
| Веб-фреймворк                 | FastAPI                                 | 0.115.6      |
| Обробка відео                 | OpenCV                                  | 4.10.0.84    |
| Математика                      | NumPy                                   | 1.26.4       |
| Машинне навчання           | PyTorch                                 | 2.3.0        |
| Асинхронність                | Uvicorn                                 | 0.34.0       |

## 💻 Вимоги

### Апаратне забезпечення:

| Компонент | Мінімальні вимоги | Рекомендовані |
| ------------------ | --------------------------------- | -------------------------- |
| CPU                | 4 ядра                        | 8+ ядер                |
| RAM                | 4 GB                              | 8+ GB                      |
| GPU                | -                                 | NVIDIA з CUDA             |
| Диск           | 2 GB                              | 10+ GB                     |
| Мережа       | 100 Mbps                          | 1 Gbps                     |

### Програмне забезпечення:

- **Python** 3.10 - 3.12
- **Операційна система**: Linux, Windows, macOS
- **Браузер**: Chrome, Firefox, Edge (для веб-інтерфейсу)
- **Кодеки**: H.264, H.265 (для RTSP)
- **CUDA**: 11.8+ (опціонально, для GPU)

### Залежності Python:

Всі залежності вказані в `requirements.txt`

## 📦 Встановлення

### 1. Клонування репозиторію

```bash
git clone https://github.com/yurikomysa/TrafficAnalysis
cd TrafficAnalysis
```

### 2. Створення віртуального середовища

```bash
# З використанням venv (Linux/Mac)
python -m venv venv
source venv/bin/activate

# З використанням venv (Windows)
python -m venv venv
venv\Scripts\activate

# З використанням conda (рекомендовано)
conda create -n traffic_analysis python=3.12
conda activate traffic_analysis
```

### 3. Встановлення залежностей

```bash
# Базове встановлення
pip install -r requirements.txt

# Для GPU підтримки (якщо є CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 4. Завантаження моделей

```bash
# Модель завантажиться автоматично при першому запуску
# Або завантажте вручну:
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### 5. Налаштування для Linux/Codespaces

```bash
# Встановіть системні залежності
sudo apt-get update
sudo apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg
```

## 🚀 Запуск

### 1. Запуск веб-сервера (FastAPI)

```bash
# Запуск в розробницькому режимі
python run.py

# Або через uvicorn
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# Для production
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 2. Запуск з командного рядка

```bash
# З відеофайлом
python -m src.main --rtsp "video.mp4"

# З RTSP камерою
python -m src.main --rtsp "rtsp://admin:password@192.168.1.100:554/stream"

# З тестовим потоком
python -m src.main --rtsp "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov"
```

### 3. Запуск з параметрами

```bash
# Повна команда з усіма параметрами
python -m src.main \
    --rtsp "rtsp://admin:123456@192.168.1.100:554/stream1" \
    --vehicle_model "yolov8n.pt" \
    --confidence 0.5 \
    --device cpu \
    --skip_frames 1 \
    --headless
```

## 🎮 Використання

### Веб-інтерфейс

1. Відкрийте браузер і перейдіть на `http://localhost:8000`
2. Введіть RTSP URL або шлях до відео
3. Натисніть **▶ Start** для початку аналізу
4. Спостерігайте за відео та статистикою в реальному часі

### Управління

| Дія       | Опис                              |
| ------------ | ------------------------------------- |
| `▶ Start` | Почати аналіз             |
| `⏹ Stop`  | Зупинити аналіз         |
| `🗑 Clear` | Очистити статистику |
| `q / ESC`  | Вихід (CLI режим)           |

### Параметри запуску

| Параметр    | Опис                                                  | За замовчуванням |
| ------------------- | --------------------------------------------------------- | ------------------------------- |
| `--rtsp`          | RTSP URL або шлях до відео                  | Обов'язковий         |
| `--vehicle_model` | Модель для детекції авто             | `yolov8n.pt`                  |
| `--confidence`    | Поріг впевненості (0-1)                   | `0.5`                         |
| `--device`        | Пристрій (`cpu` або `cuda`)                | `cpu`                         |
| `--skip_frames`   | Пропуск кадрів для оптимізації | `1`                           |
| `--headless`      | Запуск без GUI                                   | `False`                       |

## 📡 API Endpoints

### Основні endpoints

| Endpoint          | Метод | Опис                              |
| ----------------- | ---------- | ------------------------------------- |
| `/`             | GET        | Головна сторінка       |
| `/start`        | POST       | Запуск аналізу           |
| `/stop`         | POST       | Зупинка аналізу         |
| `/frame`        | GET        | Отримання кадру         |
| `/stats`        | GET        | Статистика                  |
| `/clear_stats`  | POST       | Очищення статистики |
| `/upload_video` | POST       | Завантаження відео   |
| `/health`       | GET        | Перевірка стану         |

### Приклади запитів

#### Запуск аналізу

```bash
curl -X POST http://localhost:8000/start \
  -H "Content-Type: application/json" \
  -d '{"rtsp_url": "rtsp://admin:123456@192.168.1.100:554/stream1"}'
```

#### Отримання кадру

```bash
curl http://localhost:8000/frame
```

#### Отримання статистики

```bash
curl http://localhost:8000/stats
```

## 🏗 Архітектура

### Загальна архітектура системи

```mermaid
graph TB
    subgraph "Джерела даних"
        RTSP[RTSP Camera]
        FILE[Video File]
        UPLOAD[Uploaded Video]
    end

    subgraph "Обробка"
        SR[StreamReader]
        VD[VehicleDetector]
        VT[VehicleTracker]
        CD[ColorDetector]
        PR[PlateReader]
    end

    subgraph "Веб-сервер"
        API[FastAPI Server]
        WS[WebSocket]
        STATS[Statistics]
    end

    subgraph "Інтерфейс"
        WEB[Web Interface]
        CLI[Command Line]
    end

    RTSP --> SR
    FILE --> SR
    UPLOAD --> SR
  
    SR --> VD
    VD --> VT
    VT --> CD
    VT --> PR
  
    CD --> API
    PR --> API
    VT --> API
  
    API --> WEB
    API --> CLI
    API --> WS
  
    STATS --> API
```

### Схема обробки даних

```mermaid
flowchart LR
    subgraph "Вхідні дані"
        A[RTSP Stream] --> B[Frame Buffer]
        C[Video File] --> B
    end
  
    B --> D[Frame]
  
    subgraph "Детекція"
        D --> E[YOLOv8]
        E --> F{Bounding Boxes}
    end
  
    F --> G[Tracker]
  
    subgraph "Аналіз"
        G --> H[Color Detection]
        G --> I[Plate Detection]
        H --> J[Color Name]
        I --> K[OCR]
        K --> L[Plate Text]
    end
  
    subgraph "Візуалізація"
        J --> M[Draw BBox]
        L --> M
        G --> M
        M --> N[Processed Frame]
    end
  
    N --> O[Web Interface]
    N --> P[Video Display]
```

## 🔄 Схема роботи

### Діаграма послідовності обробки кадру

```mermaid
sequenceDiagram
    participant User
    participant WebUI
    participant API
    participant StreamReader
    participant VehicleDetector
    participant Tracker
    participant ColorDetector
    participant PlateReader
    participant Display

    User->>WebUI: Start Analysis
    WebUI->>API: POST /start {rtsp_url}
    API->>StreamReader: start_processing()
  
    loop Кожен кадр
        StreamReader->>StreamReader: get_frame()
        StreamReader->>VehicleDetector: detect_vehicles(frame)
        VehicleDetector->>VehicleDetector: YOLOv8 inference
        VehicleDetector-->>Tracker: detections
        Tracker->>Tracker: update_tracks()
        Tracker-->>ColorDetector: track_id, bbox
        ColorDetector->>ColorDetector: detect_color(roi)
        ColorDetector-->>Display: color_name
        Tracker-->>PlateReader: track_id, bbox
        PlateReader->>PlateReader: detect_plates(roi)
        PlateReader->>PlateReader: read_plate(image)
        PlateReader-->>Display: plate_text
        Display->>Display: draw_vehicle_info()
        Display-->>WebUI: processed_frame
        WebUI->>User: update_video()
    end
  
    User->>WebUI: Stop Analysis
    WebUI->>API: POST /stop
    API->>StreamReader: stop_processing()
```

### Діаграма класів

```mermaid
classDiagram
    class TrafficAnalyzer {
        -StreamReader stream_reader
        -VehicleDetector vehicle_detector
        -Tracker tracker
        -ColorDetector color_detector
        -PlateReader plate_reader
        +start_processing(rtsp_url)
        +stop_processing()
        +get_frame()
        +get_stats()
        -_process_frame(frame)
        -_processing_loop()
    }

    class StreamReader {
        -string url
        -list frame_buffer
        -thread thread
        +start()
        +stop()
        +get_frame()
        +get_fps()
        -_read_loop()
        -_reconnect()
    }

    class VehicleDetector {
        -YOLO model
        -float confidence_threshold
        +detect_vehicles(frame)
        +detect_vehicles_optimized(frame)
        +get_vehicle_roi(frame, bbox)
    }

    class VehicleTracker {
        -dict tracks
        -int next_id
        +update(detections)
        +get_active_tracks()
        -_match(detections)
        -_compute_iou(detections)
    }

    class ColorDetector {
        +detect_color(roi)
        -_classify_color(hsv)
    }

    class PlateReader {
        -OCR ocr
        -Queue queue
        +detect_plates(roi)
        +read_plate(plate_id, image)
        -_preprocess(image)
        -_ocr_loop()
    }

    TrafficAnalyzer --> StreamReader
    TrafficAnalyzer --> VehicleDetector
    TrafficAnalyzer --> VehicleTracker
    TrafficAnalyzer --> ColorDetector
    TrafficAnalyzer --> PlateReader
```

## ⚡ Оптимізація

### Для edge-пристроїв

| Оптимізація | Опис                      | Ефект               |
| ---------------------- | ----------------------------- | ------------------------ |
| `skip_frames`        | Пропуск кадрів   | +50% швидкості  |
| `confidence`         | Зниження порогу | +30% детекцій    |
| `yolov8n.pt`         | Найлегша модель | +100% швидкості |
| `headless`           | Без GUI                    | -20% CPU                 |
| `resize`             | Зменшення кадру | +80% швидкості  |

### Налаштування для різних пристроїв

#### Raspberry Pi 4

```bash
python -m src.main \
    --rtsp "rtsp://..." \
    --vehicle_model "yolov8n.pt" \
    --confidence 0.4 \
    --skip_frames 3 \
    --device cpu \
    --headless
```

#### NVIDIA Jetson Nano

```bash
python -m src.main \
    --rtsp "rtsp://..." \
    --vehicle_model "yolov8n.pt" \
    --confidence 0.5 \
    --skip_frames 2 \
    --device cuda \
    --headless
```

#### Сервер з GPU

```bash
python -m src.main \
    --rtsp "rtsp://..." \
    --vehicle_model "yolov8m.pt" \
    --confidence 0.5 \
    --skip_frames 0 \
    --device cuda
```

## 🧪 Тестування

### Тестовий RTSP потік

```bash
# Безкоштовний тестовий потік
rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov
```

### Тестовий скрипт

```python
# test_system.py
import requests
import cv2
import numpy as np

# 1. Тест API
response = requests.get('http://localhost:8000/health')
print(f"Health: {response.json()}")

# 2. Тест відео
cap = cv2.VideoCapture('video.mp4')
if cap.isOpened():
    print("✅ Відео відкрито")
    ret, frame = cap.read()
    if ret:
        print(f"✅ Кадр отримано, розмір: {frame.shape}")
    cap.release()
```

## 🐛 Можливі проблеми

### 1. Помилка `libGL.so.1: cannot open shared object file`

**Рішення:**

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y libgl1

# Або використовуйте headless версію
pip install opencv-python-headless
```

### 2. Помилка RTSP 401 Unauthorized

**Рішення:**

- Додайте логін/пароль в URL: `rtsp://username:password@ip:port/stream`
- Перевірте правильність облікових даних
- Спробуйте стандартні комбінації: admin/admin, admin/123456

### 3. Повільна робота

**Рішення:**

- Зменшіть `--skip_frames` (пропускайте більше кадрів)
- Використовуйте `yolov8n.pt` (найлегша модель)
- Зменшіть роздільну здатність відео
- Використовуйте `--device cpu` якщо немає GPU

### 4. Не читаються номери

**Рішення:**

- Переконайтесь, що номер видно на відео
- Збільште контрастність відео
- Спробуйте інший `--confidence` поріг
- Перевірте чи встановлено EasyOCR

### 5. Помилка NumPy сумісності

**Рішення:**

```bash
# Встановіть правильну версію
pip install numpy==1.26.4

# Або використовуйте conda
conda install numpy=1.26.4
```

## 🔮 Плани на майбутнє

```mermaid
gantt
    title План розвитку проекту
    dateFormat  YYYY-MM-DD
    section Найближчі
    Збереження відео з розміткою    :a1, 2024-01-01, 30d
    Інтеграція з базами даних       :a2, after a1, 20d
    Розпізнавання швидкості         :a3, after a2, 25d
    section Середньострокові
    Детекція DDoS                   :b1, after a3, 30d
    Мобільний додаток              :b2, after b1, 45d
    Хмарна синхронізація           :b3, after b2, 30d
    section Довгострокові
    Telegram бот                   :c1, after b3, 20d
    Підтримка ONVIF               :c2, after c1, 30d
    Розпізнавання марки авто       :c3, after c2, 40d
```

- [ ] Додати збереження відео з розміткою
- [ ] Інтеграція з базами даних
- [ ] Розпізнавання швидкості руху
- [ ] Детекція DDoS атак
- [ ] Мобільний додаток
- [ ] Хмарна синхронізація
- [ ] Telegram бот для сповіщень
- [ ] Підтримка ONVIF протоколу
- [ ] Розпізнавання марки автомобіля
- [ ] Аналіз щільності трафіку

## 👨‍💻 Автор

Yuri Komysa - [yurikomysa@gmail.com]

## 📄 Ліцензія

Цей проект ліцензовано під MIT License - див. файл [LICENSE](LICENSE) для деталей.

## 🙏 Подяки

- [Ultralytics](https://github.com/ultralytics/ultralytics) за YOLOv8
- [FastAPI](https://fastapi.tiangolo.com/) за чудовий фреймворк
- [OpenCV](https://opencv.org/) за обробку відео
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) за розпізнавання тексту

---

## 📞 Підтримка

Якщо у вас виникли питання або проблеми:

1. Створіть Issue в GitHub
2. Напишіть на email
3. Перегляньте документацію
4. Перевірте розділ [Можливі проблеми](#-можливі-проблеми)

---

⭐ Якщо вам сподобався проект, поставте зірочку на GitHub!
