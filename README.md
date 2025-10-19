# 🧠 Tumor Detection and Diagnosis

## 📌 Overview
This project focuses on **detecting brain tumors from MRI images** using **YOLO (You Only Look Once)** — one of the fastest and most accurate object detection architectures.  
The model was trained on a **custom medical dataset** containing multiple types of brain tumors and then deployed as an **interactive Streamlit web app** for easy visualization and real-time predictions.

---

## 🎯 Project Objectives
- Build an automated detection system for **brain tumors** from MRI scans.  
- Train a **YOLO model** on custom medical image data.  
- Evaluate model performance using **Precision, Recall, and mAP**.  
- Deploy an **easy-to-use Streamlit web app** for live predictions.

---

## 🧩 Dataset
The dataset was obtained from **[Roboflow](https://roboflow.com/)** and includes MRI brain scan images categorized into **four main classes**:

| Class | Description |
|--------|--------------|
| **Glioma** | Tumor originating in glial cells |
| **Meningioma** | Tumor of the meninges (brain membranes) |
| **Pituitary** | Tumor in the pituitary gland |
| **No-Tumor** | Normal MRI (no tumor) |

### Dataset Split
- **Training set**  
- **Validation set**  
- **Testing set**

A visualization and analysis of the dataset were performed to ensure:
- Balanced class distribution  
- High-quality and accurate annotations  

---

## ⚙️ Project Workflow

### 1️⃣ Data Preparation
- Imported the dataset from **Roboflow**.  
- Verified the dataset structure:
├── train/
│ ├── images/
│ └── labels/
├── valid/
│ ├── images/
│ └── labels/
└── test/
├── images/
└── labels/
- Analyzed and visualized **class distribution** to check dataset balance.

---

### 2️⃣ Model Training
- Used the **Ultralytics YOLO** framework for model training.
- Example configuration:
```python
from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO('yolov11n.pt')

# Train on custom dataset
model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640
)
🚀 Deployment

The trained YOLO model was deployed as a Streamlit web app, allowing users to:

Upload MRI images

View tumor detection results in real-time

See bounding boxes and confidence scores for each prediction

📈 Results

The model achieved high accuracy on test data.

Key metrics such as Precision, Recall, and mAP demonstrate strong performance in detecting multiple tumor types.

🧠 Technologies Used

Python

YOLOv11 (Ultralytics)

Streamlit

Roboflow

OpenCV

NumPy / Pandas / Matplotlib
