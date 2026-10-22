# 🖐️ Sign Language Detection and Recognition System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![MediaPipe](https://img.shields.io/badge/vision-MediaPipe%20Hands-orange.svg)](https://developers.google.com/mediapipe)
[![scikit-learn](https://img.shields.io/badge/ml-scikit--learn-green.svg)](https://scikit-learn.org/)
[![Gradio](https://img.shields.io/badge/ui-Gradio%20Web-orange.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An accessible, modular, local-first computer vision application in Python that recognizes sign language fingerspelling from a standard webcam or static images and transcribes gestures into readable text in real time.

---

## 📌 Executive Overview

- **Primary Objective:** Recognize manual sign language gestures from live video or image uploads, filter out uncertain poses, smooth transient frame jitter, and accumulate recognized letters into words and sentences.
- **Target Sign Language (MVP):** **American Sign Language (ASL) Fingerspelling** (24 static alphabet letters: `A` through `Y`, excluding motion letters `J` and `Z`).
- **Core Strategy:** Extract 21 3D anatomical hand landmarks using Google MediaPipe Hands, apply coordinate centering and scale-invariant normalization, classify handshape vectors using a calibrated Machine Learning model, and process predictions through a temporal hysteresis filter with debounce cooldown.

> [!IMPORTANT]
> **Linguistic Scope & Ethical Disclaimer:**  
> This software is an **experimental educational prototype** strictly designed for isolated fingerspelling. It does **not** recognize continuous fluent sign language, grammar, or facial expressions. It must **never** be used as a replacement for certified sign language interpreters in healthcare, legal, or emergency situations.

---

## 🏗️ System Architecture

The pipeline processes input through an end-to-end decoupled workflow:

$$\text{Webcam Frame} \longrightarrow \text{MediaPipe Hands} \longrightarrow \text{Landmark Normalizer} \longrightarrow \text{ML Classifier} \longrightarrow \text{Temporal Filter} \longrightarrow \text{Text Accumulator} \longrightarrow \text{Gradio UI}$$

```
+------------------+     +-------------------+     +-------------------------+
|  Video / Camera  | --> |  MediaPipe Hands  | --> |  Coordinate Normalizer  |
|  (640x480 RGB)   |     |  (21 3D Keypoints)|     |  (Wrist Origin + Scale) |
+------------------+     +-------------------+     +-------------------------+
                                                                |
                                                                v
+------------------+     +-------------------+     +-------------------------+
|    Gradio Web    | <-- |  Text Accumulator | <-- |  Temporal Hysteresis &  |
|    Interface     |     |  (Word & Spacing) |     |  Debounce Cooldown      |
+------------------+     +-------------------+     +-------------------------+
```

### Mathematical Coordinate Normalization
To render classifications completely invariant to user distance and hand positioning within the webcam frame:
1. **Translation Invariance:** The wrist landmark ($P_0$) is subtracted from all 21 keypoints:  
   $$P'_i = P_i - P_0, \quad \forall i \in \{0, \dots, 20\}$$
2. **Scale Invariance:** All coordinates are divided by the maximum Euclidean distance from the wrist:  
   $$d_{\max} = \max_i \|P'_i\|_2, \quad P''_i = \frac{P'_i}{d_{\max} + 10^{-7}}$$
3. **Feature Vector:** The normalized 3D points are flattened into a 63-dimensional vector $\mathbf{x} \in \mathbb{R}^{63}$.

---

## 📁 Repository Structure

```
sign-language-detector/
├── .venv/                      # Isolated Python virtual environment
├── data/
│   ├── raw/                    # Raw collected landmark recordings
│   └── processed/              # Processed CSV landmark datasets
├── docs/
│   ├── PRD.md                  # Comprehensive Product Requirements Document
│   └── TRD.md                  # Detailed Technical Requirements Document
├── models/
│   ├── classifier.joblib       # Serialized Random Forest model artifact
│   ├── class_names.json        # 24 class label mapping
│   └── metadata.json           # Model parameters & validation metrics
├── reports/
│   ├── confusion_matrix.png    # Normalized confusion matrix visualization
│   └── evaluation_report.json  # Benchmark metrics, latencies, rejection rates
├── src/
│   ├── __init__.py
│   ├── config.py               # Typed application configuration dataclasses
│   ├── feature_extraction.py   # MediaPipe Hands 21-landmark extractor
│   ├── preprocessing.py        # Coordinate normalizer and augmentations
│   ├── models.py               # SignClassifier wrapper (RF & SVM)
│   ├── prediction_smoothing.py # Sliding window voting & debounce cooldown
│   ├── text_builder.py         # Sentence buffer and whitespace controls
│   ├── data_collection.py      # Interactive webcam landmark recording tool
│   ├── synthetic_data.py       # Anatomically grounded synthetic dataset builder
│   ├── train_model.py          # Training pipeline with signer-aware splits
│   ├── evaluate_model.py       # Metrics, confusion matrix, & latency benchmarks
│   └── inference.py            # Unified end-to-end inference orchestrator
├── ui/
│   ├── __init__.py
│   └── gradio_app.py           # Accessible Gradio web UI
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py   # Normalization and translation-invariance tests
│   ├── test_smoothing.py       # Majority voting and duplicate debounce tests
│   ├── test_text_builder.py    # Text formatting and backspace tests
│   ├── test_models.py          # Model loading and probability tests
│   └── test_inference.py       # End-to-end integration tests
├── app.py                      # Main CLI launcher
├── requirements.txt            # Pinned dependencies
└── README.md                   # Project documentation
```

---

## ⚡ Quick Start & Installation

### 1. Clone & Set Up Environment
```bash
# Navigate to project directory
cd sign-language-detector

# Create virtual environment
python -m venv .venv

# Activate environment
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Application

#### Option A: Interactive Web UI (Recommended)
```bash
python app.py --mode web --port 7860
```
Open your browser to: **`http://127.0.0.1:7860`**
- **Live Webcam Tab:** Start webcam, view hand skeleton overlay, observe live predictions, adjust confidence and smoothing sliders in real time.
- **Static Image Tab:** Drag and drop hand sign photos for instant classification and probability distributions.
- **Text Controls:** Append spaces, backspace characters, or clear text.

#### Option B: Native Desktop Window (Low-Latency Mode)
```bash
python app.py --mode cv2 --camera 0
```
- Press **`Space`** to insert space.
- Press **`B`** to backspace.
- Press **`C`** to clear sentence.
- Press **`L`** to toggle landmark skeleton.
- Press **`Q`** to quit.

---

## 🏋️ Training & Evaluation

### Train Model
Train a new Random Forest or SVM model on the landmark dataset:
```bash
# Train Random Forest baseline
python app.py --train --model-type random_forest

# Or train Support Vector Machine (SVM)
python app.py --train --model-type svm
```

### Run Benchmarks & Generate Reports
```bash
python app.py --evaluate
```
This generates:
- **`reports/confusion_matrix.png`**: Normalized 24-class confusion matrix.
- **`reports/evaluation_report.json`**: Top-1 Accuracy, Macro-F1, per-class metrics, P50/P95 latency, and noise rejection rate.

### Collect Custom Webcam Landmark Data
To collect real-world samples for custom signs or signers:
```bash
python src/data_collection.py --letter A --signer signer_01 --samples 50
```
- Press **`C`** or **`Space`** to toggle recording.
- Landmarks are extracted and appended to `data/raw/custom_landmarks.csv`.

---

## 🧪 Automated Testing

Execute the test suite using `pytest`:
```bash
pytest tests/ -v
```

All unit and integration tests verify:
1. Hand landmark coordinate normalization, translation invariance, and scale invariance.
2. Temporal hysteresis, sliding window consensus, and duplicate suppression debounce.
3. Text accumulator string manipulation, whitespace handling, and undo/clear operations.
4. Model serialization, probability calibration, and multi-class distribution integrity.
5. End-to-end inference execution on synthetic frames without camera dependencies.

---

## 📊 Benchmark Results

Evaluated on held-out test data across 24 ASL fingerspelling classes:

| Metric | Target | Measured Result | Status |
| :--- | :---: | :---: | :---: |
| **Top-1 Accuracy** | $\ge 88.0\%$ | **97.29%** | Passed |
| **Macro-F1 Score** | $\ge 0.85$ | **0.9729** | Passed |
| **P50 Inference Latency (CPU)** | $\le 100\text{ ms}$ | **46.50 ms** | Passed |
| **Equivalent Inference Throughput** | $\ge 15\text{ FPS}$ | **19.3 FPS** | Passed |
| **Noise & Out-of-Distribution Rejection** | $\ge 80.0\%$ | **100.0%** | Passed |

---

## 📜 Ethical Considerations

- **Biometric Privacy:** Hand landmarks and video frames are processed exclusively in volatile local RAM. Zero frames or biometric embeddings are transmitted over the internet or saved without explicit user action.
- **Fairness & Inclusion:** MediaPipe Hands operates on normalized coordinates derived from palm detection, minimizing skin-tone bias. However, lighting conditions and background contrast significantly influence landmark quality.
- **Interpreter Replacement Disclaimer:** Automated tools cannot comprehend contextual nuances, grammar, or cultural idioms of Deaf culture. This software is purely an educational demonstrator.
