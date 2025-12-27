# ASL Vision – Technical Documentation

This document provides detailed technical documentation for **ASL Vision (ASL Sign Recognizer)**, covering system architecture, data pipeline, model workflow, backend API logic, and frontend integration.

---

## 1. Project Architecture

ASL Vision is designed as a simple end-to-end pipeline:

ASL Video
↓
Frontend (HTML / CSS / JS)
↓ multipart/form-data
FastAPI Backend
↓
Video Preprocessing
↓
PyTorch Model Inference
↓
Prediction + Confidence + Top-5
↓
Frontend Visualization

yaml
Copy code

The system is optimized for **local usage and demos**, keeping components loosely coupled and easy to debug.

---

## 2. Dataset

### Dataset Used

**WLASL Processed Dataset**

Source:
https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed

### Why WLASL

- Widely used in academic ASL research
- Preprocessed video samples
- Supports large-scale sign classification

### Dataset Role in This Project

- Training and fine-tuning the ASL recognition model
- Defining class labels used during inference
- Benchmarking model behavior (no fixed accuracy claimed)

---

## 3. Data Pipeline

### 3.1 Data Collection

- Videos are collected from the WLASL processed dataset
- Each video corresponds to a labeled ASL sign

### 3.2 Exploratory Data Analysis (EDA)

Performed to:

- Inspect class distribution
- Identify class imbalance
- Analyze video length and frame count

### 3.3 Preprocessing

Typical preprocessing steps include:

- Video decoding
- Frame sampling (temporal subsampling)
- Frame resizing and normalization
- Conversion to model-compatible tensor format

> ⚠️ Preprocessing at inference time must match training preprocessing.

---

## 4. Model Overview

### Model Type

- Deep learning model implemented in **PyTorch**
- Trained and fine-tuned on ASL video data

### Model Input

- Short video clip
- Fixed number of frames after sampling
- Normalized pixel values

### Model Output

- Probability distribution over sign classes
- Top-1 prediction (highest probability)
- Top-5 ranked predictions

### Model Files

- Stored as `.pth` checkpoints
- Loaded at backend startup or on demand
- Path configurable via environment variable

---

## 5. Backend (FastAPI)

### 5.1 Responsibilities

- Accept video uploads
- Validate input format
- Run preprocessing and inference
- Return structured JSON results

### 5.2 API Endpoints

#### `GET /ping`

Purpose:

- Health check
- Used by frontend to show Online / Offline status

Response:

```json
```json
{ "status": "ok" }
```

## POST /predict

Request:

Content-Type: multipart/form-data

Field name: file

Value: ASL video (mp4 or webm)

##### Response:

```

{
  "prediction": "HELLO",
  "confidence": 0.87,
  "top5": [
    { "label": "HELLO", "confidence": 0.87 },
    { "label": "THANKS", "confidence": 0.05 },
    { "label": "PLEASE", "confidence": 0.03 }
  ]
}
```

### 5.3 Environment Configuration

Optional `.env` file inside `backend/`:

<pre class="overflow-visible! px-0!" data-start="3107" data-end="3152"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-env"><span>MODEL_PATH=../models/asl_model.pth
</span></code></div></div></pre>

If not provided, the backend uses its internal default model path.

## 6. Frontend

### 6.1 Technology

---

* HTML (structure)
* CSS (styling and layout)
* Vanilla JavaScript (logic and API calls)

### 6.2 Frontend Responsibilities

* Allow users to upload a video
* Preview the selected video
* Check backend availability
* Send video to backend
* Display prediction results

### 6.3 Backend Status Indicator

* Frontend calls `GET /ping`
* Displays:
  * **Online** if backend responds correctly
  * **Offline** otherwise

### 6.4 Result Visualization

Displayed elements:

* Predicted label
* Confidence percentage
* Ordered Top-5 predictions list

---

## 7. Error Handling

### Common Issues and Causes

* **CORS errors**
  * Frontend origin not allowed in FastAPI CORS settings
* **422 Validation Error**
  * Incorrect form field name (must be `file`)
  * Request not sent as `multipart/form-data`
* **Unsupported video format**
  * Use `mp4` (H.264) or compatible `webm`
* **Backend Offline**
  * FastAPI server not running
  * Wrong API base URL in frontend

---

## 8. Performance Notes

* Inference runs on CPU by default
* GPU is optional but not required
* Short clips provide faster and more stable predictions
* Longer videos should be trimmed before upload

---

## 9. Team Responsibilities

* **Seif Magdy**

  GUI, Backend, Frontend–API Integration
* **Mohab Shahin**

  Model Training and Experimentation
* **Mohamed Elmosalamy**

  Data EDA, Collection, Preprocessing, Fine-tuning
* **Ammar Amged**

  Training and Fine-tuning
* **Mohamed Saed Fayed**

  GUI Development and Model Fine-tuning

---

## 10. Future Improvements

* Streaming / real-time inference
* Improved temporal modeling
* Model optimization and pruning
* Dockerized deployment
* Automated testing for API and preprocessing

---

## 11. Notes

This documentation focuses on  **system design and integration** .

Detailed training scripts, hyperparameters, and experiments can be documented separately if needed.

---
