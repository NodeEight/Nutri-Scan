import os
import io
import torch
import logging
from torch import nn
from PIL import Image
from torchvision import transforms


from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "model/malnutrition_model_5wk2l.pt")


# =====================
# FastAPI App
# =====================
app = FastAPI()

# Allow CORS for all origins (for testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for API call count and model accuracy
api_call_count = 0

# Set device for inference
device = "cuda" if torch.cuda.is_available() else "cpu"


# Model definition (must match the one used during training)
class MalnutritionModel(nn.Module):
    def __init__(self):
        super(MalnutritionModel, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=2, stride=2, padding=0
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.activation = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64 * 50 * 50, 2)  # Output 2 classes

    def forward(self, data):
        data = self.conv1(data)
        data = self.activation(data)
        data = self.maxpool(data)
        data = self.activation(data)
        data = self.flatten(data)
        data = self.linear(data)
        return data


# Image preprocessing (must match training pipeline)
data_trans = transforms.Compose([transforms.Resize((200, 200)), transforms.ToTensor()])


# Load model at startup
def load_model():
    model = MalnutritionModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model


model_instance = load_model()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Receives an image file, returns prediction, API call count, and model accuracy.
    """
    global api_call_count
    api_call_count += 1
    try:
        # Read image bytes
        contents = await file.read()
        predicted_label, model_accuracy = model_inference(model_instance, contents)

        return JSONResponse(
            {
                "prediction": predicted_label,
                "api_call_count": api_call_count,
                "model_accuracy": model_accuracy,
            }
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/")
def root():
    return {
        "message": "Malnutrition Model API. Use /predict to POST an image and get Prediction."
    }


def model_inference(model, image_bytes):
    """
    Run inference on a single image file with safety checks and confidence scores.
    """
    # 1. Set model to evaluation mode
    model.eval()

    # 2. Load and Transform
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    # Ensure the transform resizes to 200x200 to match your model's hardcoded Linear layer
    # If data_trans is defined globally, make sure it includes transforms.Resize((200, 200))
    input_image = data_trans(image).unsqueeze(0).to(device)

    # 3. Inference
    with torch.no_grad():
        output = model(input_image)

        # Calculate probabilities using Softmax
        probs = F.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probs, 1)

    # 4. Map Labels (Assuming 1=Malnourished, 0=Nourished from your training fix)
    class_label_map = {1: "Malnourished", 0: "Nourished"}
    predicted_label = class_label_map.get(predicted_class.item(), "Unknown")

    logger.info(f"Predicted: {predicted_label} | Confidence: {confidence.item():.2%}")

    return predicted_label, float(confidence.item()) * 100
