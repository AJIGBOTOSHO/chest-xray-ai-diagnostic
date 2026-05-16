# 🫁 Explainable AI Diagnostic Assistant for Chest X‑rays

A web‑based tool that classifies chest X‑rays into **Normal, Pneumonia, Tuberculosis, and COVID‑19** — and visually explains *where* the model focused using Grad‑CAM heatmaps and bounding‑box localisation. Trained on real Nigerian patient data from **Aminu Kano Teaching Hospital**.

![Demo](https://img.icons8.com/color/96/lungs.png)

## 🚀 Live Demo
Try the app now: **[Streamlit Cloud URL](https://chest-xray-ai-diagnostic.streamlit.app/)**

---

## 📌 Table of Contents
- [Problem Statement](#-problem-statement)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Model Performance](#-model-performance)
- [How It Works](#-how-it-works)
- [Getting Started (Local)](#-getting-started-local)
- [Deployment](#-deployment)
- [Challenges & Solutions](#-challenges--solutions)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 🔴 Problem Statement
Most chest X‑ray AI models are trained on Western populations and fail on African patients. Even when accurate, they are **black‑box** – clinicians cannot see *why* a prediction was made. This project addresses both gaps with a locally‑trained, explainable diagnostic assistant.

---

## ✅ Features
- 🔍 **4‑class classification** – Normal, Pneumonia, Tuberculosis, COVID‑19
- 🧠 **Grad‑CAM heatmaps** – shows which image regions influenced the decision
- 📦 **Bounding‑box localisation** – automatically draws a box around the area of interest
- 📋 **AI explanation summary** – plain‑language clinical note + recommendation
- 📊 **Confidence bar chart** – full probability distribution
- ⬇️ **Downloadable report** – save predictions as a `.txt` file
- ⚙️ **Adjustable sensitivity** – tune heatmap opacity and bounding‑box threshold

---

## 🧰 Tech Stack
| Component | Technology |
|-----------|------------|
| Deep Learning | TensorFlow / Keras |
| Architecture | DenseNet121 (transfer learning) |
| Explainability | Grad‑CAM, bounding‑box thresholding |
| Preprocessing | CLAHE (contrast enhancement) |
| Web Framework | Streamlit |
| Deployment | Streamlit Community Cloud |
| Language | Python 3.11 |

---

## 📊 Dataset
**Nigeria Chest X‑Ray Dataset** – 2,600 annotated images (650 per class) from Aminu Kano Teaching Hospital.  
Available on [Kaggle](https://www.kaggle.com/).

---

## 📈 Model Performance
| Metric | Value |
|--------|-------|
| Architecture | DenseNet121 (fine‑tuned) |
| Test Accuracy | **96.9%** |
| Classes | Normal, Pneumonia, TB, COVID‑19 |
| Input Size | 224×224 RGB (CLAHE‑enhanced) |

---

## ⚙️ How It Works
1. **Upload** a chest X‑ray image (JPG/PNG).
2. **Preprocessing** – CLAHE contrast enhancement + resize to 224×224.
3. **Prediction** – DenseNet121 outputs class probabilities.
4. **Explainability**:
   - Grad‑CAM heatmap is computed from the last convolutional layer.
   - Heatmap is thresholded to extract a bounding box around the region of interest.
   - Overlay is displayed alongside the original image and raw heatmap.
5. **Report** – AI summary table and downloadable text report.

---

## 🖥 Getting Started (Local)

### Prerequisites
- Python 3.11+ (3.12 also works with `tensorflow-cpu>=2.18`)
- Git (optional)

### Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/chest-xray-ai-diagnostic.git
cd chest-xray-ai-diagnostic

# Create & activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt 