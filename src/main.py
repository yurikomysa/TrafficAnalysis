from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import os
import cv2
import numpy as np
import base64
from typing import Optional
import shutil
import json

from .traffic_analyzer import TrafficAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Traffic Analysis API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Створюємо папки
os.makedirs("static", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# Монтуємо статичні файли
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    logger.warning(f"Static files mount error: {e}")

# Глобальні змінні
analyzer = None
is_running = False


def convert_numpy_types(obj):
    """Конвертує NumPy типи в стандартні Python типи для JSON"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(v) for v in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


@app.on_event("startup")
async def startup_event():
    global analyzer
    logger.info("🚗 Starting Traffic Analysis API")
    try:
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        os.environ['DISPLAY'] = ':0'
        os.environ['OPENCV_OPENCL_RUNTIME'] = ''
        
        analyzer = TrafficAnalyzer(confidence=0.5, device='cpu', skip_frames=1)
        logger.info("✅ Analyzer initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize analyzer: {e}")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Головна сторінка"""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
        <head><title>Traffic Analysis</title></head>
        <body style="font-family: Arial; background: #0a0a1a; color: white; padding: 40px; text-align: center;">
            <h1 style="color: #00ff88;">🚗 Traffic Analysis System</h1>
            <p style="color: #888;">Please create static/index.html or use the API directly.</p>
            <div style="margin-top: 30px;">
                <h3>API Endpoints:</h3>
                <p><code>POST /start</code> - Start analysis</p>
                <p><code>POST /stop</code> - Stop analysis</p>
                <p><code>GET /frame</code> - Get current frame</p>
                <p><code>GET /stats</code> - Get statistics</p>
            </div>
            <form action="/start" method="post" style="margin-top: 30px;">
                <input type="text" name="rtsp_url" value="video.mp4" style="padding:10px;width:300px;border-radius:5px;border:1px solid #333;background:#1a1a2e;color:white;">
                <button type="submit" style="padding:10px 30px;background:#00ff88;color:#0a0a1a;border:none;border-radius:5px;cursor:pointer;font-weight:bold;">▶ Start</button>
            </form>
            <a href="/stop" style="display:inline-block;margin-top:10px;padding:10px 30px;background:#ff4444;color:white;text-decoration:none;border-radius:5px;">⏹ Stop</a>
        </body>
        </html>
        """)


@app.post("/start")
async def start_analysis(data: dict):
    """Запуск аналізу"""
    global analyzer, is_running
    
    rtsp_url = data.get('rtsp_url')
    if not rtsp_url:
        raise HTTPException(status_code=400, detail="rtsp_url is required")
    
    if is_running:
        return {"status": "error", "message": "Analysis already running"}
    
    if not analyzer:
        analyzer = TrafficAnalyzer(confidence=0.5, device='cpu', skip_frames=1)
    
    success = analyzer.start_processing(rtsp_url)
    
    if success:
        is_running = True
        return {"status": "success", "message": f"Analysis started for {rtsp_url}"}
    else:
        return {"status": "error", "message": "Failed to start analysis"}


@app.post("/stop")
async def stop_analysis():
    """Зупинка аналізу"""
    global is_running, analyzer
    is_running = False
    if analyzer:
        analyzer.stop_processing()
    return {"status": "success", "message": "Analysis stopped"}


@app.get("/frame")
async def get_frame():
    """Отримання останнього кадру"""
    global analyzer
    if not analyzer or not is_running:
        return JSONResponse({"status": "error", "message": "Analysis not running"})
    
    result = analyzer.get_frame()
    if result and result.get('frame'):
        frame_b64 = base64.b64encode(result['frame']).decode('utf-8')
        
        # Конвертуємо дані в JSON-сумісний формат
        vehicles = convert_numpy_types(result.get('vehicles', []))
        
        return JSONResponse({
            "status": "success",
            "frame": frame_b64,
            "vehicles": vehicles,
            "count": int(result.get('count', 0)),
            "fps": float(result.get('fps', 0))
        })
    
    return JSONResponse({"status": "error", "message": "No frame available"})


@app.get("/stats")
async def get_stats():
    """Отримання статистики"""
    global analyzer
    if not analyzer:
        return JSONResponse({"status": "error", "message": "Analyzer not initialized"})
    
    stats = analyzer.get_stats()
    if stats:
        # Конвертуємо всі NumPy типи
        stats = convert_numpy_types(stats)
        return JSONResponse(stats)
    
    return JSONResponse({"status": "error", "message": "No stats available"})


@app.post("/clear_stats")
async def clear_stats():
    """Очищення статистики"""
    global analyzer
    if analyzer:
        analyzer.plates_detected = 0
        analyzer.vehicle_info.clear()
    return {"status": "success"}


@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """Завантаження відеофайлу"""
    if not file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Unsupported video format")
    
    file_path = os.path.join("uploads", file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {
        "status": "success",
        "filename": file.filename,
        "path": file_path,
        "url": f"/uploads/{file.filename}"
    }


@app.get("/health")
async def health_check():
    """Перевірка стану"""
    return {
        "status": "running",
        "analyzer": analyzer is not None,
        "is_running": is_running,
        "vehicles": int(analyzer.detected_vehicles) if analyzer else 0,
        "plates": int(analyzer.plates_detected) if analyzer else 0
    }


def run_server(host="0.0.0.0", port=8000):
    """Запуск сервера"""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()