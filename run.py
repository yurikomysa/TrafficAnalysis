#!/usr/bin/env python
import sys
import os
import uvicorn

# Додаємо src в шлях
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Встановлюємо змінні середовища для headless режиму
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['DISPLAY'] = ':0'
os.environ['OPENCV_OPENCL_RUNTIME'] = ''
os.environ['MPLBACKEND'] = 'Agg'

if __name__ == "__main__":
    print("=" * 60)
    print("🚗 Traffic Analysis API Server")
    print("=" * 60)
    print("📍 Open in browser: http://localhost:8000")
    print("📡 API Docs: http://localhost:8000/docs")
    print("=" * 60)
    print("")
    
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )