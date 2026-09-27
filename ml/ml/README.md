# ♻️ AI Smart Waste Management System

> **Real-Time Waste Classification Using Deep Learning (MobileNetV2), FastAPI, OpenCV, and SQLite**

An end-to-end intelligent waste sorting and computer vision application that classifies waste items in real time via live webcam or image upload into **6 canonical categories**:
1. **Plastic** (Blue Bin)
2. **Paper** (Cyan Bin)
3. **Metal** (Yellow Bin)
4. **Glass** (Teal Bin)
5. **Organic** (Green Compost Bin)
6. **General Waste** (Black Bin)

---

## 🌟 Key Features

- **Real-Time Computer Vision**: Point your webcam at an item to receive live frame-by-frame AI predictions.
- **Deep Learning Inference**: Uses lightweight, CPU-optimized **MobileNetV2** (ONNX Runtime) combined with OpenCV material analysis (specular reflection, HSV chrominance, edge density).
- **Smart Bin Guidance**: Instantly recommends the appropriate disposal bin with colour coding, disposal instructions, and eco facts.
- **Duplicate Debouncing**: Intelligent 5-second SQLite logging debounce prevents redundant database entries while holding an item.
- **Live Analytics Dashboard**: Real-time KPI counters, category distribution breakdown, dynamic doughnut charts, and one-click CSV export.
- **Zero-GPU Required**: Fast, responsive inference engineered to run smoothly on any standard student laptop or PC.
- **RESTful API & Swagger**: Interactive OpenAPI documentation at `/docs` with automatic Pydantic validation.

---

## 🛠️ Architecture & Tech Stack

| Layer | Technology |
|---|---|
| **Backend Framework** | FastAPI (Python 3.12+) & Uvicorn ASGI |
| **Deep Learning** | MobileNetV2 ONNX Runtime (CPU) |
| **Computer Vision** | OpenCV (`cv2`) & Pillow (`PIL`) |
| **Frontend UI** | HTML5, CSS3 Glassmorphism, Vanilla JS |
| **Database** | SQLite3 (WAL mode) |
| **Data Validation** | Pydantic v2 |

---

## 🚀 Quick Start & How to Run

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python run.py
```
Or directly using Uvicorn:
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 4. Access the Application
Open your web browser and navigate to:
- **Live AI Scanner Dashboard**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Waste Analytics**: [http://127.0.0.1:8000/analytics](http://127.0.0.1:8000/analytics)
- **Interactive Swagger API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **System Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 📂 Project Structure

```
d:/ml/
├── app/
│   ├── __init__.py           # Package initialization
│   ├── config.py             # System configuration, paths, thresholds
│   ├── database.py           # SQLite connection, logging, analytics queries
│   ├── main.py               # FastAPI routes & API endpoints
│   ├── model.py              # MobileNetV2 ONNX & OpenCV material fusion
│   ├── schemas.py            # Pydantic request & response models
│   └── waste_rules.py        # 6-class bin rules, colors, instructions
├── data/
│   └── waste.db              # Local SQLite database (auto-generated)
├── model/
│   ├── mobilenetv2.onnx      # Pretrained ONNX model weights
│   └── imagenet_classes.txt  # Class label index
├── static/
│   ├── css/
│   │   └── style.css         # Modern responsive glassmorphism styles
│   └── js/
│       ├── app.js            # Webcam capture & real-time UI updates
│       └── analytics.js      # Charts & telemetry dashboard logic
├── templates/
│   ├── index.html            # Main scanner dashboard template
│   └── analytics.html        # Analytics & insights page template
├── .env                      # Environment variables
├── requirements.txt          # Python dependencies
├── run.py                    # Entry point runner script
└── README.md                 # Project documentation
```

---

## 📡 REST API Summary

- `GET /` — Real-Time AI Camera Dashboard
- `GET /analytics` — Live Analytics & Telemetry Page
- `POST /api/predict` — Multipart image upload / frame classification
- `POST /api/predict-base64` — Base64 webcam frame classification
- `GET /api/rules` — Retrieve 6-category disposal rules and bin specs
- `GET /api/stats` — Real-time detection aggregates & distributions
- `GET /api/detections` — Retrieve recent detection history
- `DELETE /api/detections` — Clear SQLite history
- `GET /api/model-info` — Inspect active AI model specifications
- `GET /health` — Check server and database health
