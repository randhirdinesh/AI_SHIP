
# Biofouling Detection Production System

A production-grade system for real-time biofouling detection and classification using deep learning. This project leverages state-of-the-art models (YOLOv8, EfficientNetV2, ConvNeXt), advanced data augmentation, and production optimizations (quantization, TensorRT conversion, asynchronous inference) to deliver robust performance in real-time environments.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Monitoring and Performance](#monitoring-and-performance)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This repository implements a full production pipeline for detecting and classifying biofouling in images and video streams. It includes:

- A custom dataset loader (`BiofoulingDataset`) with advanced augmentation (including mixup) to boost training on limited data.
- Production-ready model initialization and inference (`ProductionModel`), which ensembles detections from YOLOv8 with classifications from EfficientNetV2 and ConvNeXt.
- Optimizations for real-time performance, including quantization, optional TensorRT conversion, and asynchronous processing using threading and queues.
- Real-time monitoring and logging of performance metrics via MLflow.

---

## Features

- **Real-Time Inference:** Designed for high throughput with low latency (<33ms target per frame) to support real-time video processing.
- **Advanced Data Augmentation:** Uses Albumentations and mixup techniques to improve model robustness on limited datasets.
- **Model Ensemble:** Combines object detection (YOLOv8) and classification (EfficientNetV2, ConvNeXt) using a weighted ensemble for improved accuracy.
- **Production Optimizations:** 
  - Quantization for both CPU and GPU environments.
  - Optional conversion to TensorRT for further performance boosts on CUDA-enabled devices.
- **Asynchronous Processing:** Uses multi-threaded queues to decouple image processing from inference, ensuring smooth real-time performance.
- **Performance Monitoring:** Integrates MLflow for tracking inference times, FPS, and resource utilization.

---

## Project Structure

```
├── models
│   ├── production/                # Production model files
│   ├── yolov8s.pt                 # YOLOv8 weights file
│   ├── efficientnetv2_rw_s.pth    # EfficientNetV2 weights file
│   └── convnext_small.pth         # ConvNeXt weights file
├── cache/                         # Directory for caching images
├── logs/                          # Log files for production monitoring
├── data
│   ├── images/                    # Directory containing input images
│   └── labels/                    # Directory containing YOLO-format labels
├── README.md                      # This file
└── main.py                        # Entry point containing all classes and main() function
```

- **BiofoulingDataset:** Handles image and label loading with caching and augmentation.
- **ProductionConfig:** Centralizes configuration parameters (image size, batch size, optimization flags, etc.) and sets up logging and directories.
- **ProductionModel:** Loads, optimizes, and ensembles multiple models for detection and classification.
- **ProductionDeployment:** Manages real-time video processing, performance monitoring, and system deployment.

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/biofouling-detection.git
   cd biofouling-detection
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages:**

   ```bash
   pip install -r requirements.txt
   ```

   Make sure your `requirements.txt` includes dependencies such as:
   - OpenCV (`opencv-python`)
   - PyTorch and torchvision
   - timm
   - ultralytics
   - albumentations
   - scikit-learn
   - mlflow
   - onnx & onnxruntime
   - torch2trt (if using TensorRT conversion)

4. **Setup additional dependencies:**
   - For GPU support, ensure CUDA is properly installed.
   - If you wish to use TensorRT conversion, install `torch2trt` and ensure your environment supports it.

---

## Usage

### Running the Production System

To start the production system and process a video stream (defaulting to the primary webcam):

```bash
python main.py
```

- **Real-Time Video:** The system will capture video frames, run inference, and display detection and classification results with performance metrics overlayed.
- **Stopping:** Press `q` in the video window to exit the application gracefully.

### Synchronous vs. Asynchronous Inference

- **Asynchronous Mode (default):** The system queues frames for processing, ideal for real-time video.
- **Synchronous Mode:** Modify the `predict` method call (by passing `async_mode=False`) in the code if you require synchronous inference.

---

## Configuration

All configurable parameters are centralized in the `ProductionConfig` class:

- **Image & Video Settings:** Image size, batch size, and confidence/NMS thresholds.
- **Optimization Flags:** Toggle quantization, TensorRT conversion, and mixed precision.
- **Hardware Settings:** Specify number of threads, CUDA device visibility, and power mode (especially for Jetson platforms).
- **Paths:** Directories for models, cache, and logs are automatically created if they do not exist.

Customize these parameters by modifying the `ProductionConfig` class in `main.py` or by extending the class to load configurations from an external file.

---

## Monitoring and Performance

- **MLflow Integration:** The system logs performance metrics (inference latency, FPS, GPU utilization, and memory usage) to an MLflow tracking server. The default tracking URI is set to a local SQLite database.
- **Real-Time Alerts:** The monitoring thread continuously checks for low FPS, high latency, or excessive memory usage and logs warnings accordingly.

---
