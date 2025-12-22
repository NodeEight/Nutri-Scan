from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os

# Initialize App
app = FastAPI(
    title="Nutri-Scan API",
    description="API for detecting malnutrition in children using specific body part images.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODELS_DIR = "models"
BODY_PARTS = ['back', 'body', 'finger', 'head', 'leg', 'muac', 'side']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Store loaded models
models_dict = {}

def get_model_architecture():
    """Returns the base model architecture"""
    # Use weights=None to avoid warnings and just get architecture
    model = models.resnet18(weights=None) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model

def load_models():
    """Load all available models for body parts"""
    loaded_count = 0
    for part in BODY_PARTS:
        model_path = os.path.join(MODELS_DIR, f"nutriscan_model_{part}.pth")
        if os.path.exists(model_path):
            try:
                model = get_model_architecture()
                if torch.cuda.is_available():
                    map_location = None
                else:
                    map_location = torch.device('cpu')
                
                model.load_state_dict(torch.load(model_path, map_location=map_location))
                model = model.to(device)
                model.eval()
                models_dict[part] = model
                print(f"✅ Loaded model for: {part}")
                loaded_count += 1
            except Exception as e:
                print(f"❌ Error loading model for {part}: {e}")
        else:
            print(f"⚠️ Model file not found for {part} at {model_path}")
    return loaded_count

@app.on_event("startup")
async def startup_event():
    print("Loading models...")
    count = load_models()
    print(f"Startup complete. {count}/{len(BODY_PARTS)} models loaded.")

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.get("/")
async def root():
    return {
        "message": "Welcome to Nutri-Scan API", 
        "available_body_parts": list(models_dict.keys())
    }

@app.post("/predict")
async def predict(
    body_part: str = Query(..., description=f"Body part to analyze. Options: {', '.join(BODY_PARTS)}"),
    file: UploadFile = File(...)
):
    # Validate body part
    body_part = body_part.lower()
    if body_part not in BODY_PARTS:
        raise HTTPException(status_code=400, detail=f"Invalid body part. Must be one of: {BODY_PARTS}")
    
    # Get model
    model = models_dict.get(body_part)
    if not model:
        raise HTTPException(status_code=503, detail=f"Model for '{body_part}' is not loaded or available.")
    
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JPEG or PNG image.")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
        # Class mapping
        # 0: Malnourished, 1: Normal
        classes = ["Malnourished", "Nourished"] 
        result = classes[predicted_class.item()]
        conf_score = confidence.item()
        
        return JSONResponse(content={
            "body_part": body_part,
            "prediction": result,
            "confidence": f"{conf_score:.2f}",
            "is_nourished": bool(predicted_class.item() == 1)
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
