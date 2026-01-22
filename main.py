import io
import json
import os
from typing import List, Optional, Union
import logging


import httpx
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from dotenv import load_dotenv

from schema.models import PredictionRequest, PredictionAPIResponse
from utils.agent import agent

load_dotenv()

log = logging.getLogger("nouritrack_api")
log.setLevel(logging.INFO)


# Initialize App
app = FastAPI(
    title="Nouritrack API",
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
        "message": "Welcome to Nouritrack API", 
        "available_body_parts": list(models_dict.keys())
    }


async def fetch_and_inference(url, body_part, model):
    """Downloads a single image and opens it."""
    try:
        print(f"Starting download: {url}")
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            if response.status_code != 200:
                print(f"Failed to download {url}: Status code {response.status_code}")
                return {"status": 400, "detail": f"Failed to download image from {url}"}

        # Convert bytes to image
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        
        # Display image information and open it
        print(f"Successfully downloaded {url} ({img.format})")

        # Preprocess
        input_tensor = transform(img).unsqueeze(0).to(device)
        
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
        
        return {
            "body_part": body_part,
            "prediction": result,
            "confidence": f"{conf_score:.2f}",
            "is_nourished": bool(predicted_class.item() == 1)
        }
            
    except Exception as e:
        print(f"Failed to download {url}: {e}")


@app.post("/predict", status_code=200, response_model=PredictionAPIResponse)
async def predict(request: PredictionRequest):
    """Endpoint to predict malnutrition status from body part images and generate diagnostic report.
    <hr>
    frontView -> leg
    sideProfile -> side
    faceCloseUp -> head
    arm -> muac
    
    """
    
    results = []
    for item in request.body_parts:
        body_part, image_url = item.body_part.lower(),  item.image_url
        print(f"Processing body part: {body_part} with URL: {image_url}")
        
        # Validate body part
        if body_part not in BODY_PARTS:
            ret = {"body_part": body_part,
                   "error": f"Invalid body part. Must be one of: {BODY_PARTS}"}
            results.append(ret)
            continue
    
        # Get model
        model = models_dict.get(body_part)
        if not model:
            log.error(f"Model for '{body_part}' is not loaded or available.")
            ret = {"body_part": body_part,
                   "error": f"Model for '{body_part}' is not loaded or available."}
            results.append(ret)
            continue
     
        try:
            result = await fetch_and_inference(image_url, body_part, model)
            results.append(result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    try:
        print("Invoking agent for diagnostic report generation...")
        agent_response = agent.invoke({"messages": [{"role": "user", "content": request.model_dump_json() }]})
        resp = agent_response["structured_response"].model_dump_json()
        print(f"Diagnostic report: {resp}.")
        results.append({"diagnostic_report": json.loads(resp)})
        print("Agent invocation successful.")
    except Exception as e:
        print(f"Agent invocation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Agent invocation failed: {str(e)}")
    
    return JSONResponse(content={"status_code": 200, "results": results})

