# 🧱 AI Wall Crack Detector

A deep learning-based web application that detects cracks in wall surfaces from images using **Transfer Learning** with MobileNetV2. Upload a wall image and get an instant prediction — **Crack Detected** or **No Crack** — along with a confidence score.

---

## 🚀 Demo

Upload any wall image via the web interface and receive a real-time prediction powered by a fine-tuned MobileNetV2 model.

---

## 🧠 How It Works

1. A **MobileNetV2** model pretrained on ImageNet is used as the base (transfer learning)
2. The top layers are replaced with a custom binary classification head (Dense + Sigmoid)
3. The model is trained on labeled wall images (cracked vs non-cracked)
4. A **Flask** web app exposes a `/predict` endpoint that accepts image uploads and returns predictions

**Model Architecture:**
```
MobileNetV2 (frozen, pretrained on ImageNet)
    → GlobalAveragePooling2D
    → Dense(1, activation='sigmoid')
```

**Training Config:**
- Input size: 224×224 RGB
- Loss: Binary Crossentropy
- Optimizer: Adam
- Epochs: 5
- Batch size: 32

---

## 📁 Project Structure

```
ai-wall-crack-detector/
│
├── train/               # Training images (crack / no_crack)
├── validation/          # Validation images
├── test/                # Test images
├── templates/           # HTML templates for Flask UI
│
├── train.py             # Model training script
├── predict.py           # Standalone prediction script
├── app.py               # Flask web application
├── wall_crack_model.h5  # Saved trained model
├── requirements.txt     # Python dependencies
└── README.md
```

---

## ⚙️ Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/ksrikarsai/ai-wall-crack-detector.git
cd ai-wall-crack-detector
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the web app**
```bash
python app.py
```

Then open `http://localhost:5000` in your browser.

---


## 🛠️ Tech Stack

| Technology | Usage |
|---|---|
| Python | Core language |
| TensorFlow / Keras | Model training & inference |
| MobileNetV2 | Pretrained base model (Transfer Learning) |
| Flask | Web API and UI |
| NumPy | Image array processing |
| Pillow (PIL) | Image loading and preprocessing |

---

## 📊 API Reference

**POST** `/predict`

| Parameter | Type | Description |
|---|---|---|
| `file` | image file | Wall image (JPEG/PNG) |

**Response:**
```json
{
  "prediction": "Crack Detected",
  "score": 0.23
}
```

- Score closer to `0` → Crack Detected
- Score closer to `1` → No Crack

---

## 🏗️ Real-World Use Case

Manual inspection of walls for structural cracks is time-consuming and error-prone. This tool automates the detection process, useful for:
- Construction site safety checks
- Building maintenance systems
- Infrastructure monitoring pipelines

---

## 📦 Requirements

```
tensorflow
flask
numpy
pillow
```

Install with:
```bash
pip install -r requirements.txt
```

---

