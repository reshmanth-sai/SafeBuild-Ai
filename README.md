# SafeBuild AI 🦺🤖

**SafeBuild AI** is an AI-powered computer vision & safety analytics platform designed for real-time Personal Protective Equipment (PPE) compliance monitoring, danger zone intrusion detection, and workplace safety incident reporting.

## 🌟 Key Features

- **PPE Detection & Tracking**: Real-time object detection and tracking using YOLOv8 (`Hexmon/vyra-yolo-ppe-detection`) fine-tuned for 14 safety classes (Hardhat, Safety Vest, Gloves, Masks, NO-PPE violations, Fall Detection, etc.).
- **Multi-Object Tracking**: Custom ByteTrack configuration (`custom_bytetrack.yaml`) with high frame persistence for reliable worker tracking.
- **Incident & Intrusion Detection**: Automated detection of danger zone intrusions and non-compliance with debounced incident reporting.
- **FastAPI Backend**: Robust API layer (`backend/main.py`) with SQLite database (`backend/safety.db`) to record incidents, worker safety events, and analytics.
- **Interactive Dashboards**:
  - `dashboard.html`: Live safety overview, active alerts, and PPE compliance statistics.
  - `rescue_dashboard.html`: Emergency rescue & worker location monitoring.
  - `simulator.html`: Test and simulate safety breach scenarios.
  - `timeline.html`: Interactive incident timeline and historical analysis.

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- OpenCV, PyTorch, Ultralytics YOLO, FastAPI, Uvicorn, SQLite3

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ReshmanthSai/SafeBuild-Ai.git
   cd SafeBuild-Ai
   ```

2. **Set up a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install ultralytics opencv-python numpy fastapi uvicorn huggingface_hub
   ```

### Running the Application

1. **Start the Backend Server:**
   ```bash
   uvicorn backend.main:app --reload
   ```

2. **Run PPE Webcam Detection:**
   ```bash
   python day1_ppe_webcam.py
   ```

3. **View Dashboards:**
   Open `dashboard.html` in any modern web browser.

## 📁 Repository Structure

```
├── backend/
│   ├── main.py          # FastAPI application & API endpoints
│   ├── database.py      # SQLite database connection & setup
│   ├── models.py        # SQLAlchemy models
│   └── schemas.py       # Pydantic data schemas
├── custom_bytetrack.yaml # ByteTrack tracker configuration
├── day1_ppe_webcam.py   # YOLOv8 PPE detection & tracking script
├── dashboard.html       # Main safety monitoring dashboard
├── rescue_dashboard.html# Emergency rescue dashboard
├── simulator.html       # Breach simulation page
├── timeline.html        # Historical incident timeline
├── README.md            # Project documentation
└── .gitignore           # Git ignore configuration
```

## 📄 License

MIT License
