# Video Faces Extract 🎥👤

A Python-based tool for detecting, tracking, and extracting faces from videos using multiple algorithms. Tracks face appearances and generates metadata for searchable face datasets.

## Features ✨
- **Multi-Algorithm Support**:  
  - `MediaPipe` (blazing-fast detection)  
  - `face-recognition` (high accuracy)  
  - `FastFace` (lightweight model)
- **Face Tracking**: Track faces across video frames and calculate appearance duration.
- **Output Generation**:  
  - CSV with face timestamps (start/end time, duration)  
  - Extracted face images for search/identification
- **Flexible Configuration**: Adjust detection thresholds, frame sampling, and tracking parameters.

---

## Installation 📦

### Dependencies
```bash
pip install opencv-python numpy pandas face-recognition fastface mediapipe
```

## Usage 🚀

### Basic Video Processing
```python
from algo import FaceDetectionTimeTracker
from config import VIDEO_PATH

# Initialize with MediaPipe detector
tracker = FaceDetectionTimeTracker(model="mediapipe")
results = tracker.from_video(
    VIDEO_PATH,
    n_sec=2,               # Analyze every 2 seconds
    memorize_face_sec=5,   # Track faces for 5s after disappearance
    detect_threshold=0.7   # Confidence threshold
)

# Results contain face metadata + images
print(f"Detected {len(results)} unique faces")
```

### Output Structure
- CSV Columns:
  - start_index_frame, end_index_frame, duration_existence, encoding, face_image

- Saved Images:
  - High-quality face crops stored in SAVE_IMAGE_PATH (configured in config.py)

## Algorithms Comparison 🔍

| Algorithm        | Speed       | Accuracy   | Best For               |
|------------------|-------------|------------|------------------------|
| **MediaPipe**    | ⚡️⚡️⚡️      | Medium     | Real-time processing   |
| **face-recognition** | ⚡️        | 🎯 High    | Precision tasks        |
| **FastFace**     | ⚡️⚡️        | Medium     | Low-resource devices   |

## Configuration ⚙️

Update `config.py` with your custom paths:

```python
# Path to MediaPipe face detection model
MODEL_MEDIAPIPE_PATH = 'path/to/detector.tflite'  # Required for MediaPipe detector

# Video processing paths
VIDEO_PATH = 'path/to/input_video.mp4'          # Input video file
SAVE_IMAGE_PATH = 'path/to/output_faces/'       # Directory for extracted faces
```
