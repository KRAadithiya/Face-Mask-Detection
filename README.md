# Face Mask Detection System
### Final Year Project | Python · OpenCV · TensorFlow/Keras · Flask

---

## Project Overview
An end-to-end Face Mask Detection system that uses **MobileNetV2** (transfer learning) for classification and a **ResNet-10 SSD** for face detection. Supports real-time webcam inference, image file analysis, and video file processing.

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download face detector models
Place these two files inside the `models/` folder:
- `deploy.prototxt`
- `res10_300x300_ssd_iter_140000.caffemodel`

Download from OpenCV's GitHub:
```bash
# deploy.prototxt
wget https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt -P models/

# caffemodel (~10 MB)
wget https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel -P models/
```

### 3. Prepare dataset
```bash
# Verify structure
python src/dataset_prep.py --verify

# Check for corrupt images
python src/dataset_prep.py --integrity

# Optional: balance classes via augmentation
python src/dataset_prep.py --augment --target 3000
```

Dataset layout expected:
```
dataset/
  with_mask/      ← ~3700 images
  without_mask/   ← ~3800 images
```
> Download from: https://www.kaggle.com/datasets/omkargurav/face-mask-dataset

### 4. Train the model
```bash
python src/train_model.py
```
This saves:
- `models/mask_detector.h5` — trained model
- `models/training_plot.png` — accuracy/loss curves
- `models/confusion_matrix.png`

---

## Run Detection

### Webcam (real-time)
```bash
python src/detect_mask.py --mode webcam
```
Press `q` to quit.

### Image file
```bash
python src/detect_mask.py --mode image --input path/to/image.jpg
```

### Video file
```bash
python src/detect_mask.py --mode video --input path/to/video.mp4
```

---

## Launch Web Dashboard
```bash
python src/app.py
```
Open http://localhost:5000 in your browser.

Features:
- Drag-and-drop image upload with live annotation
- Browser webcam detection (frames sent to backend)
- Real-time detection stats and compliance bar charts
- Detection log with per-face confidence scores

---

## Project Structure
```
face_mask_detection/
├── src/
│   ├── train_model.py        # Model training
│   ├── detect_mask.py        # CLI detection
│   ├── app.py                # Flask web dashboard
│   └── dataset_prep.py       # Dataset helpers
├── models/                   # Weights + detector files
├── dataset/                  # Training images
├── templates/index.html      # Web UI
├── requirements.txt
├── README.md
└── Face_Mask_Detection_Report.docx
```

---

## Tech Stack
| Component        | Technology                        |
|------------------|-----------------------------------|
| Language         | Python 3.8+                       |
| Deep Learning    | TensorFlow 2.x / Keras            |
| Base Model       | MobileNetV2 (ImageNet weights)    |
| Face Detection   | OpenCV DNN (ResNet-10 SSD)        |
| Web Framework    | Flask                             |
| Image Processing | OpenCV, NumPy, Pillow             |
| Training Utilities | scikit-learn, matplotlib, seaborn |

---

## Model Performance
| Metric    | Value  |
|-----------|--------|
| Accuracy  | ~98.4% |
| Precision | ~0.98  |
| Recall    | ~0.98  |
| F1-Score  | ~0.98  |
