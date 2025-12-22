# Nutri-Scan Deep Learning Model & API

This project contains deep learning modules for detecting child malnutrition from images of specific body parts using PyTorch and FastAPI. The system employs a **Ensemble of Specialized Models** strategy, where individual models are trained for specific anatomical regions to maximize diagnostic accuracy.

## Project Structure
- `train_model.py`: Training script that generates specialized ResNet18 models for each body part.
- `api.py`: FastAPI application serving the specialized models via a unified endpoint.
- `requirements_ml.txt`: Python dependencies.
- `models/`: Directory where trained models (`.pth` files) are saved.

---

## 🧠 Training Architecture

### 1. Strategy: Specialized Body Part Models
Instead of a single monolithic model, we train separate binary classifiers for each anatomical region. This reduces noise and allows the model to learn features specific to that body part (e.g., rib visibility for 'body' vs. face shape for 'head').

**Supported Body Parts:**
- Back
- Body (General Torso/Ribs)
- Finger
- Head
- Leg
- MUAC (Mid-Upper Arm Circumference)
- Side

### 2. Model Backbone: ResNet-18 (Transfer Learning)
We use **ResNet-18** as the backbone for each specialized model.
- **Pre-trained Weights**: The models are initialized with weights pre-trained on ImageNet. This allows the model to leverage learned feature extractors (edges, textures) despite the potentially small size of the medical dataset.
- **Custom Head**: The final fully connected layer of ResNet-18 is replaced with a new linear layer: `nn.Linear(num_ftrs, 2)`. This outputs logits for the two classes: **Malnourished** (0) and **Nourished** (1).

### 3. Data Preprocessing & Augmentation
To improve generalization and prevent overfitting, the following transformations are applied during training:

**Training Transforms:**
- **Resize**: Images are resized to `224x224` pixels (standard input for ResNet).
- **Random Horizontal Flip**: Simulates mirror images.
- **Random Rotation**: ±15 degrees to handle variations in camera angle.
- **Color Jitter**: Randomly adjusts brightness (0.1) and contrast (0.1) to account for different lighting conditions.
- **Normalization**: Standard ImageNet normalization (Mean: `[0.485, 0.456, 0.406]`, Std: `[0.229, 0.224, 0.225]`).

**Validation Transforms:**
- **Resize**: 224x224.
- **Normalization**: Standard ImageNet normalization.

### 4. Hyperparameters
- **Optimizer**: Adam (Adaptive Moment Estimation)
- **Learning Rate**: 0.001
- **Batch Size**: 16
- **Epochs**: 500 (Extended training for maximum convergence)
- **Loss Function**: CrossEntropyLoss

---

## Setup & Usage

### 1. Install Dependencies
```bash
pip install -r requirements_ml.txt
```

### 2. Data Organization
Ensure your dataset is structured as follows. The training script recursively searches these folders, so exact subfolder names inside `head/`, `back/`, etc., don't matter, but the top-level body part folders must exist.

```
Nutri-Scan/
├── malnourished/
│   ├── head/
│   ├── back/
│   ├── body/
│   ├── finger/
│   ├── leg/
│   ├── muac/
│   └── side/
└── normal/
    ├── head/
    ├── back/
    └── ... (same structure)
```

### 3. Training
Run the training script to sequentially train models for all body parts.
```bash
python train_model.py
```
*Models will be saved to the `models/` directory.*

### 4. Running the API
Start the FastAPI server. It will load all available models from the `models/` directory on startup.
```bash
uvicorn api:app --reload
```
*Access API at: `http://127.0.0.1:8000`*

---

## API Documentation

### POST `/predict`
Analyzes an uploaded image using the model specific to the provided `body_part`.

**Parameters:**
- `body_part` (Query Param): The anatomical region of the image. Options: `head`, `back`, `body`, `finger`, `leg`, `muac`, `side`.
- `file` (Form Data): The image file (JPEG/PNG).

**Response:**
```json
{
    "body_part": "head",
    "prediction": "Malnourished",
    "confidence": "0.98",
    "is_nourished": false
}
```
