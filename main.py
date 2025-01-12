from flask import Flask, Response, render_template_string
from ultralytics import YOLO
import supervision as sv
import cv2
import wiringpi
from wiringpi import GPIO
import sys
import time
from collections import deque
import threading

# Источник ввода видео
input_camera = 1

# Инициализация Flask приложения
app = Flask(__name__)

# Инициализация GPIO
wiringpi.wiringPiSetup()
GPIO_PIN = 2  # Номер GPIO пина
wiringpi.pinMode(GPIO_PIN, GPIO.OUTPUT)  # Установка режима вывода для пина

# Модель НС
ncnn_model = YOLO('yolo11n_ncnn_model/', task='detect')

# Экземпляры классов трекера и аннотаторов
tracker = sv.ByteTrack()
tracker.reset()  # Обнуляем ID объектов трекера
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)

# Открываем видеопоток с камеры
cap = cv2.VideoCapture(input_camera)
if not cap.isOpened():
    print("Ошибка: Не удается открыть камеру.")
    sys.exit(1)

# Устанавливаем разрешение 640x480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Глобальная переменная для хранения последнего кадра
latest_frame = None
frame_lock = threading.Lock()

def process_video_stream():
    global latest_frame
    prev_time = time.time()  # Время предыдущего кадра
    fps_values = deque(maxlen=30)  # Очередь для хранения последних 10 значений FPS

    while True:
        success, frame = cap.read()  # Читаем кадр из видеопотока
        if not success:
            break
            
        frame = cv2.flip(frame, flipCode=1)

        results = ncnn_model(frame, verbose=False, imgsz=160, conf=0.4, agnostic_nms=True)[0]

        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)
        labels = [
            f"#{tracker_id} {results.names[class_id]}"
            for class_id, tracker_id
            in zip(detections.class_id, detections.tracker_id)
        ]
        frame = box_annotator.annotate(frame.copy(), detections)
        frame = label_annotator.annotate(frame, detections, labels)

        # Подсчет FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Добавляем FPS в очередь и вычисляем среднее
        fps_values.append(fps)
        avg_fps = sum(fps_values) / len(fps_values)
        avg_fps_rounded = int(round(avg_fps))
        print(avg_fps_rounded)

        # Отображение FPS на изображении
        cv2.putText(
            frame, f"FPS: {avg_fps_rounded}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
            (0, 255, 0), 2, cv2.LINE_AA)
        
        # Обновляем последний кадр
        with frame_lock:
            latest_frame = frame

def generate_frames():
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        # Кодируем кадр в JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Генерируем байты в формате MJPEG
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    # Возвращаем поток изображений
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    # Возвращаем HTML-страницу с кнопкой и видео
    return render_template_string('''
    <html>
        <head>
            <title>YOLO on Orange Pi</title>
            <style>
                @font-face {
                    font-family: 'Roboto';
                    src: url('static/Roboto-Regular.ttf') format('truetype');
                }
                body {
                    text-align: center;
                }
                h1 {
                    font-family: 'Roboto', sans-serif;
                    font-size: 26px;
                    color: #333;
                    margin: 20px 0; /* Отступы сверху и снизу по 20px */
                }
                button {
                    font-family: 'Roboto', sans-serif;
                    font-size: 20px;
                    padding: 12px 24px;
                    margin: 20px;
                    border: none;
                    border-radius: 8px; /* Закругленные углы */
                    cursor: pointer;
                    transition: all 0.3s ease; /* Плавный переход для эффектов */
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); /* Тень */
                }
                button.off {
                    background-color: gray;
                    color: white;
                }
                button.on {
                    background-color: green;
                    color: white;
                }
                button:hover {
                    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15); /* Увеличенная тень при наведении */
                    transform: translateY(-2px); /* Лёгкий "подъём" */
                }
                button:active {
                    transform: translateY(1px); /* Лёгкое нажатие */
                    box-shadow: 0 3px 6px rgba(0, 0, 0, 0.1); /* Более маленькая тень */
                }
            </style>

        </head>
        <body>
            <h1>Видеопоток с камеры</h1>
            <img src="/video_feed">
            <div>
                <button id="gpioButton" class="off" onclick="toggleGPIO()">Включить</button>
            </div>
            <script>
                function updateButton(state) {
                    const button = document.getElementById('gpioButton');
                    if (state === 'HIGH') {
                        button.textContent = 'Отключить';
                        button.className = 'on';
                    } else {
                        button.textContent = 'Включить';
                        button.className = 'off';
                    }
                }

                function toggleGPIO() {
                    fetch('/toggle_gpio')
                        .then(response => response.json())
                        .then(data => {
                            updateButton(data.state);
                        });
                }

                // Инициализация кнопки при загрузке страницы
                fetch('/gpio_state')
                    .then(response => response.json())
                    .then(data => {
                        updateButton(data.state);
                    });
            </script>
        </body>
    </html>
    ''')

@app.route('/toggle_gpio')
def toggle_gpio():
    current_state = wiringpi.digitalRead(GPIO_PIN)  # Текущее состояние GPIO

    # Переключаем состояние GPIO
    new_state = GPIO.HIGH if current_state == GPIO.LOW else GPIO.LOW
    wiringpi.digitalWrite(GPIO_PIN, new_state)
    
    return {'state': 'HIGH' if new_state == GPIO.HIGH else 'LOW'}  # Возвращаем новое состояние

@app.route('/gpio_state')
def gpio_state():
    current_state = wiringpi.digitalRead(GPIO_PIN)  # Текущее состояние GPIO
    return {'state': 'HIGH' if current_state == GPIO.HIGH else 'LOW'}  # Возвращаем новое состояние

if __name__ == '__main__':
    try:
        # Запускаем обработку видеопотока в отдельном потоке
        video_thread = threading.Thread(target=process_video_stream)
        video_thread.daemon = True
        video_thread.start()

        # Запускаем Flask-приложение
        app.run(host='0.0.0.0', port=5000)  # Запускаем веб-сервер
    except KeyboardInterrupt:
        print("Сервер остановлен пользователем!")
    finally:
        cap.release()  # Освобождаем ресурсы камеры
        wiringpi.digitalWrite(GPIO_PIN, GPIO.LOW)  # Выключаем GPIO перед выходом
