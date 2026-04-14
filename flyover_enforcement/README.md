# Flyover Enforcement System 🚦

**Automated Two-Wheeler Detection & Fine Generation for Restricted Flyovers**

> A production-ready computer vision system that detects two-wheelers illegally entering a restricted flyover/highway using CCTV footage, reads their number plates via ANPR, and auto-generates fine reports.

**Location:** Kerala, India (Palarivattom Flyover, Kochi)

---

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [DIP Techniques Used](#dip-techniques-used)
- [Setup Instructions](#setup-instructions)
- [Running the System](#running-the-system)
- [Project Structure](#project-structure)
- [Module Details](#module-details)
- [Dashboard](#dashboard)
- [Testing](#testing)
- [Configuration](#configuration)

---

## 🏗 Architecture Overview

```
CCTV Feed → Stabilization → Night/Day Detection → Enhancement → Noise Removal
    ↓
ROI Masking → Motion Detection → Vehicle Detection (YOLOv8) → Motion Filter
    ↓
Two-Wheeler Found? → Plate Detection → Plate DIP → OCR → Validation
    ↓
Violation? → Deduplication → SQLite Log → PDF Report → Email/SMS Notification
    ↓
Streamlit Dashboard (Live feed, Stats, Manual Review, DIP Debug)
```

### System Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         CCTV INPUT                              │
│                  (Video File / RTSP Stream)                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING STAGE                           │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐ │
│  │ Stabilizer   │ │ Night Detect │ │ Enhancement Pipeline     │ │
│  │ (Optical     │ │ (Brightness  │ │ • Gamma Correction       │ │
│  │  Flow LK)    │ │  Threshold)  │ │ • CLAHE (LAB L-channel)  │ │
│  │              │ │              │ │ • Bilateral Denoise      │ │
│  │              │ │              │ │ • Unsharp Mask           │ │
│  │              │ │              │ │ • Dark Channel Dehaze    │ │
│  └──────────────┘ └──────────────┘ └──────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DETECTION STAGE                              │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐ │
│  │ ROI Manager  │ │ MOG2 Motion  │ │ YOLOv8 Vehicle Detector  │ │
│  │ (Polygon     │ │ Filter       │ │ • motorcycle (class 3)   │ │
│  │  Masking)    │ │ (BG Subtract)│ │ • bicycle (class 1)      │ │
│  └──────────────┘ └──────────────┘ │ • car/bus/truck (allowed) │ │
│                                     └──────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                        ANPR STAGE                                │
│  ┌──────────────┐ ┌──────────────────┐ ┌──────────────────────┐ │
│  │ Plate        │ │ Plate DIP        │ │ PaddleOCR Engine     │ │
│  │ Detector     │ │ • Resize         │ │ • Text Recognition   │ │
│  │ (YOLOv8 /    │ │ • Grayscale      │ │ • Kerala Regex       │ │
│  │  Contour     │ │ • Bilateral      │ │   KL\d{2}[A-Z]+\d{4}│ │
│  │  Fallback)   │ │ • Deskew (Hough) │ │ • OCR Error Fix      │ │
│  │              │ │ • Otsu Binarize  │ │   O→0, I→1, S→5     │ │
│  │              │ │ • Morph Clean    │ │                      │ │
│  │              │ │ • Pad Border     │ │                      │ │
│  └──────────────┘ └──────────────────┘ └──────────────────────┘ │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    VIOLATION & REPORTING                          │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐ │
│  │ Logic Engine │ │ Deduplicator │ │ PDF Report Generator     │ │
│  │ (Rule-based  │ │ (5-min       │ │ (ReportLab + QR Code)    │ │
│  │  Evaluation) │ │  Cooldown)   │ │                          │ │
│  ├──────────────┤ ├──────────────┤ ├──────────────────────────┤ │
│  │ SQLite       │ │ Email/SMS    │ │ Streamlit Dashboard      │ │
│  │ Logger       │ │ Notifier     │ │ (Stats, Review, Export)  │ │
│  └──────────────┘ └──────────────┘ └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔬 DIP Techniques Used

### Preprocessing (Night & Day)
| Technique | Module | Purpose | OpenCV Function |
|-----------|--------|---------|-----------------|
| Gaussian Blur | `noise_removal.py` | General smoothing | `cv2.GaussianBlur()` |
| Bilateral Filter | `noise_removal.py` | Edge-preserving denoise (primary for night) | `cv2.bilateralFilter()` |
| Non-Local Means | `noise_removal.py` | Heavy grain removal | `cv2.fastNlMeansDenoisingColored()` |
| Median Blur | `noise_removal.py` | Salt-and-pepper noise | `cv2.medianBlur()` |
| CLAHE | `enhancement.py` | Local contrast enhancement (LAB L-channel) | `cv2.createCLAHE()` |
| Histogram EQ | `enhancement.py` | Global contrast stretch | `cv2.equalizeHist()` |
| Gamma Correction | `enhancement.py` | Power-law brightness control (LUT) | `cv2.LUT()` |
| Unsharp Masking | `enhancement.py` | Edge detail recovery | `cv2.addWeighted()` |
| Kernel Sharpening | `enhancement.py` | High-pass convolution | `cv2.filter2D()` |
| Dark Channel Prior | `enhancement.py` | Fog/haze removal (He et al.) | Custom implementation |
| Optical Flow | `stabilizer.py` | Camera shake stabilization | `cv2.calcOpticalFlowPyrLK()` |
| Affine Warp | `stabilizer.py` | Frame transformation | `cv2.warpAffine()` |

### Detection
| Technique | Module | Purpose | Implementation |
|-----------|--------|---------|----------------|
| MOG2 Background Subtraction | `motion_filter.py` | Moving object detection | `cv2.createBackgroundSubtractorMOG2()` |
| Morphological Open/Close | `motion_filter.py` | Motion mask cleanup | `cv2.morphologyEx()` |
| Polygon Masking | `roi_manager.py` | ROI definition | `cv2.fillPoly()` |
| Point-in-Polygon | `roi_manager.py` | Detection filtering | `cv2.pointPolygonTest()` |

### ANPR (Number Plate Processing)
| Technique | Module | Purpose | Implementation |
|-----------|--------|---------|----------------|
| Canny Edge Detection | `plate_detector.py` | Plate boundary finding | `cv2.Canny()` |
| Contour Analysis | `plate_detector.py` | Plate localization (fallback) | `cv2.findContours()` |
| Hough Line Transform | `plate_dip.py` | Skew angle detection | `cv2.HoughLinesP()` |
| Affine Rotation | `plate_dip.py` | Deskewing | `cv2.getRotationMatrix2D()` |
| Otsu Thresholding | `plate_dip.py` | Optimal binarization | `cv2.threshold(THRESH_OTSU)` |
| Adaptive Threshold | `plate_dip.py` | Fallback binarization | `cv2.adaptiveThreshold()` |
| Erosion + Dilation | `plate_dip.py` | Character cleanup | `cv2.erode()`, `cv2.dilate()` |

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.10+
- pip package manager
- (Optional) NVIDIA GPU + CUDA for faster YOLOv8 inference

### Installation

```bash
# 1. Navigate to project directory
cd flyover_enforcement

# 2. Create virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux/Mac:
# source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download YOLOv8 weights (auto-downloads on first run)
# The system uses yolov8n.pt by default (auto-downloaded by ultralytics)

# 5. Create required directories
mkdir -p data/raw_footage data/violations data/samples models
```

### Docker Setup (Optional)
```bash
docker build -t flyover-enforcement .
docker run -p 8501:8501 flyover-enforcement
```

---

## ▶ Running the System

### Process a Video File
```bash
python -m src.pipeline --source data/raw_footage/test.mp4 --output data/violations/output.mp4
```

### Process Live CCTV Stream
```bash
python -m src.pipeline --source rtsp://camera_ip:554/stream
```

### Launch Dashboard
```bash
streamlit run src/dashboard/app.py
```

### Run Tests
```bash
pytest tests/ -v
```

### Command Line Options
```bash
python -m src.pipeline --help
  --config    Path to settings.yaml (default: config/settings.yaml)
  --source    Video file or RTSP URL
  --output    Output video path (optional)
  --no-display  Run headless (no display window)
```

---

## 📁 Project Structure

```
flyover_enforcement/
│
├── config/
│   ├── settings.yaml              # Thresholds, ROI, camera, notifications
│   └── vehicle_classes.yaml       # COCO class mapping (restricted/allowed)
│
├── data/
│   ├── raw_footage/               # Input CCTV clips
│   ├── samples/                   # Sample frames for DIP testing
│   └── violations/                # Snapshots, SQLite DB, reports
│
├── models/                        # YOLOv8 weight files
│
├── src/
│   ├── preprocessing/
│   │   ├── noise_removal.py       # Gaussian, bilateral, NLMeans, median
│   │   ├── enhancement.py         # CLAHE, histogram EQ, gamma, sharpen, dehaze
│   │   ├── night_mode.py          # Night-specific DIP pipeline
│   │   └── stabilizer.py          # Optical flow video stabilization
│   │
│   ├── detection/
│   │   ├── vehicle_detector.py    # YOLOv8 vehicle detection + classification
│   │   ├── roi_manager.py         # Polygon ROI masking
│   │   └── motion_filter.py       # MOG2 background subtraction
│   │
│   ├── anpr/
│   │   ├── plate_detector.py      # YOLO + contour plate localization
│   │   ├── plate_dip.py           # Plate DIP pipeline (critical module)
│   │   └── ocr_engine.py          # PaddleOCR + Kerala regex validation
│   │
│   ├── violation/
│   │   ├── logic_engine.py        # Violation evaluation rules
│   │   ├── deduplicator.py        # Per-plate cooldown window
│   │   └── logger.py              # SQLite violation storage
│   │
│   ├── reporting/
│   │   ├── pdf_generator.py       # ReportLab PDF fine reports
│   │   └── notifier.py            # Email (SMTP) + SMS (Twilio)
│   │
│   ├── dashboard/
│   │   └── app.py                 # Streamlit web dashboard
│   │
│   └── pipeline.py                # Main orchestrator
│
├── tests/
│   ├── test_dip.py                # DIP technique tests
│   ├── test_anpr.py               # ANPR pipeline tests
│   └── test_violation.py          # Violation logic tests
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 📊 Module Details

### Night Mode Pipeline (`night_mode.py`)
The most DIP-intensive module, designed for Kerala's challenging low-light CCTV conditions:

1. **Gamma Correction** (γ=1.8) → lifts shadow regions
2. **Bilateral Filter** (d=9) → removes noise, preserves plate edges
3. **CLAHE** (clip=3.0) → enhances local contrast on L-channel
4. **Unsharp Mask** (strength=1.5) → recovers detail softened by denoising
5. **Dark Frame Subtraction** → removes fixed-pattern sensor noise (optional)

### Plate DIP Pipeline (`plate_dip.py`)
Critical 7-step preprocessing for OCR accuracy:

1. **Resize** → 64px height, maintain aspect ratio
2. **Grayscale** → single channel for processing
3. **Bilateral Denoise** → edge-preserving smoothing (d=5)
4. **Deskew** → Hough lines detect skew angle → affine rotation
5. **Binarize** → Otsu + adaptive threshold fallback
6. **Morphological Clean** → erode + dilate character blobs
7. **Pad** → white border around plate for OCR

### Kerala Plate Validation
Pattern: `KL` + `2 digits` + `1-2 letters` + `4 digits`
Example: `KL07BJ4545`, `KL01A1234`
Regex: `r'^KL\d{2}[A-Z]{1,2}\d{4}$'`

With OCR error correction: `O→0`, `I→1`, `S→5`, `B→8`, `G→6`

---

## 🖥 Dashboard

Launch with: `streamlit run src/dashboard/app.py`

### Pages:
- **Dashboard:** Statistics cards, live feed status, recent violations
- **Violations:** Sortable/filterable table, CSV export
- **Manual Review:** Approve/reject flagged violations with snapshots
- **DIP Debug:** Upload frames, toggle DIP effects, side-by-side comparison
- **Settings:** View current configuration, database info

---

## ✅ Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_dip.py -v

# Run with coverage
pytest tests/ -v --cov=src
```

### Test Coverage:
- **test_dip.py:** Shape preservation, contrast improvement, night detection, binarization
- **test_anpr.py:** Plate detection, OCR cleaning, Kerala validation, error correction
- **test_violation.py:** Logic engine rules, deduplication cooldown, SQLite CRUD

---

## ⚙ Configuration

Edit `config/settings.yaml` to customize:

| Setting | Default | Description |
|---------|---------|-------------|
| `camera.fps_process` | 5 | Process every Nth frame |
| `thresholds.detection_confidence` | 0.45 | YOLO confidence threshold |
| `thresholds.ocr_confidence` | 0.70 | Minimum OCR confidence |
| `thresholds.night_brightness_threshold` | 80 | Mean brightness for night detection |
| `thresholds.violation_cooldown_seconds` | 300 | Per-plate dedup cooldown (5 min) |
| `fine.amount` | 500 | Fine amount in INR |
| `fine.location` | Palarivattom Flyover | Violation location name |
| `roi.entry_polygon` | [[100,400],...] | Flyover entry zone coordinates |

---

## 📄 License

This project is developed for academic/research purposes as part of a Digital Image Processing course project.

---

## 👥 Authors

Built for Kerala Motor Vehicles Department (Academic Prototype)

**DIP Course Project — Two-Wheeler Flyover Enforcement System**
