# 🏭 Computer Vision for Real-Time Analytics: MobileNetV2 for Manufacturing Quality Control

## 🚀 Live Demo & Repository Structure

| Component | Description | Status |
| :--- | :--- | :--- |
| **Live Application** | [Link to your Hugging Face Space/Streamlit Cloud URL] | **[LIVE URL](https://huggingface.co/spaces/Swathi04/Casting-Defect-QC-MobileNetV2)** |
| **Model Artifact** | `final_best_model.pt` | Saved |
| **Code Structure** | `app.py`, `inference_handler.py` | Complete |

---

## 💡 Project Overview: Simulating Quality Control (QC)

This project demonstrates an end-to-end Computer Vision pipeline designed to replace manual visual inspection in a manufacturing setting. It utilizes **Transfer Learning** on a lightweight model (**MobileNetV2**) to achieve low-latency predictions, suitable for deployment as a **"real-time"** quality control gate.

### **Business Problem**
In high-volume manufacturing, visual inspection for defects is time-consuming and prone to human error. This system automatically classifies casting products as **Defective** or **OK**.

### **Key Features**
1.  **High-Speed Inference:** Uses MobileNetV2 for fast, low-latency prediction.
2.  **Transfer Learning:** Efficient training by leveraging ImageNet pre-trained weights.
3.  **Explainability (Grad-CAM):** Provides a heatmap overlay  to show **where** the model is focusing, building trust and helping quality engineers pinpoint physical defect locations.

---

## 🧠 Technical Implementation

### **1. Dataset**
* **Source:** Casting Product Image Data for Quality Inspection (Kaggle).
* **Task:** Binary Image Classification (`Defective` vs. `OK`).
* **Input:** Single-channel grayscale images ($300 \times 300$).

### **2. Model Architecture & Training**
* **Base Model:** **MobileNetV2** (pre-trained on ImageNet).
* **Technique:** **Transfer Learning**.
    1.  **Feature Extraction:** Loaded MobileNetV2 base and **froze all layers**. Trained a new classification head (Sigmoid output) for 5 epochs.
    2.  **Fine-Tuning:** **Unfroze the last 3 convolutional blocks** and the classification head. Trained the entire model with a very low learning rate ($10^{-5}$) for 5 additional epochs.

### **3. Deployment & UI Stack**
* **Inference Backend:** PyTorch and `inference_handler.py`.
* **Web Framework:** **Gradio** (`app.py`) for rapid, interactive deployment.
* **Deployment Target:** Hugging Face Spaces (or Streamlit Cloud).

---

## 📊 Performance Results (Test Set)

The model was optimized for **Recall**, as missing a true defect (**False Negative**) is the highest cost error in a quality control scenario.

| Metric | Value | Interpretation (QC Focus) |
| :--- | :--- | :--- |
| **Recall (Defect Class)** | **[Insert Recall Value]** | Percentage of *actual* defects successfully caught by the model. |
| **Precision** | **[Insert Precision Value]** | Percentage of 'Defect' predictions that were correct. |
| **F1-Score** | **[Insert F1-Score Value]** | Harmonic mean of Precision and Recall. |
| **Accuracy** | **[Insert Accuracy Value]** | Overall percentage of correct predictions. |

### **Real-Time Latency**
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Average Latency** | **[Insert Latency in ms] ms** | Time taken for a single prediction (critical for 'real-time'). |
| **Effective Frame Rate** | **[Insert FPS Value] FPS** | The maximum rate at which the model can process images. |

---

## ⚙️ How to Run Locally

### **1. Clone the Repository**
```bash
git clone [Your Repository URL]
cd [project-folder]
