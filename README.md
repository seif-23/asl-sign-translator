# ASL Vision (ASL Sign Recognizer)

![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)

**ASL Vision** is a local end-to-end American Sign Language (ASL) sign recognition demo.
Upload a short video clip (`mp4` / `webm`) and receive a predicted sign label, confidence score, and top-5 results through a FastAPI backend and a lightweight static frontend.

## **Used Data**

WLASL is the largest video dataset for Word-Level American Sign Language (ASL) recognition, which features 2,000 common different words in ASL.  WLASL will facilitate the research in sign language understanding and eventually benefit the communication between deaf and hearing 

communities.

#### Acknowledgements

All the WLASL data is intended for academic and computational use only. No commercial usage is allowed.

Made by Dongxu Li and Hongdong Li. Please read [the WLASL paper](https://arxiv.org/abs/1910.11006) and visit the official [website](https://dxli94.github.io/WLASL/) and [repository](https://github.com/dxli94/WLASL).

Licensed under the Computational Use of Data Agreement (C-UDA). Please [refer to the C-UDA-1.0 page](https://github.com/microsoft/Computational-Use-of-Data-Agreement/releases/tag/v1.0) for more information.

---

## Overview

This project integrates:

- A **FastAPI backend** for deep learning inference
- A **static HTML / CSS / JavaScript frontend** for video upload, preview, and result visualization

It is intended for **local experiments, academic projects, and demos**, focusing on clarity and full pipeline integration.

---

## Features

- Video upload via `multipart/form-data`
- Top-1 and Top-5 ASL sign predictions
- Backend status check using `GET /ping`
- Video preview before submission
- Clear UI output (prediction, confidence %, top-5 list)

---

## Dataset

This project uses the **WLASL Processed Dataset**:

https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed

---

## Requirements

- Python 3.9+
- pip
- Windows / macOS / Linux
- GPU not required (CPU inference supported)

---

## Installation

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r backend/requirements.txt
```


## Run


### Backend


```
cd backend
univcorn main:app --host127.0.0.1 --port8000 --reload
```



API base:

```
http://127.0.0.1:8000
```


### Frontend


Open directly:

```
frontend/index.html
```



## API

### `GET /ping`

<pre class="overflow-visible! px-0!" data-start="2063" data-end="2093"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-json"><span><span>{</span><span></span><span>"status"</span><span>:</span><span></span><span>"ok"</span><span></span><span>}</span><span>
</span></span></code></div></div></pre>

### `POST /predict`

* **Field:** `file`
* **Type:** `multipart/form-data`

Example response:

<pre class="overflow-visible! px-0!" data-start="2188" data-end="2358"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-json"><span><span>{</span><span>
  </span><span>"prediction"</span><span>:</span><span></span><span>"HELLO"</span><span>,</span><span>
  </span><span>"confidence"</span><span>:</span><span></span><span>0.87</span><span>,</span><span>
  </span><span>"top5"</span><span>:</span><span></span><span>[</span><span>
    </span><span>{</span><span></span><span>"label"</span><span>:</span><span></span><span>"HELLO"</span><span>,</span><span></span><span>"confidence"</span><span>:</span><span></span><span>0.87</span><span></span><span>}</span><span>,</span><span>
    </span><span>{</span><span></span><span>"label"</span><span>:</span><span></span><span>"THANKS"</span><span>,</span><span></span><span>"confidence"</span><span>:</span><span></span><span>0.05</span><span></span><span>}</span><span>
  </span><span>]</span><span>
</span><span>}</span><span>
</span></span></code></div></div></pre>

---

## Team

* **Seif Magdy** — GUI, Backend and inference
* **Mohab Shahin** — Training , inference
* **Mohamed Elmosalamy** — Data EDA, Collection, Fine-tuning
* **Ammar Amged** — Training, Fine-tuning,preprocessing
* **Mohamed Saed Fayed** — GUI, Fine-tuning

---

## Notes

* Model checkpoints are loaded from `models/` as `.pth` files.
* Accuracy varies by dataset and training configuration.
* Detailed preprocessing, training, and evaluation steps are documented separately.

---

## License

No license specified yet. Add a `LICENSE` file if needed.

---

## Acknowledgements

* FastAPI & Uvicorn
* PyTorch
* WLASL dataset contributors

