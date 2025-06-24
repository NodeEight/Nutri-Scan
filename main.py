from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io
import torch
from torch import nn
from PIL import Image
from torchvision import transforms

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
model_accuracy = 0.9930*100 # Set your real accuracy here

# Set device for inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model path
# trained_model_path = r'C:\Users\TechWatt\Desktop\techwatt\Nutri-Scan\model\malnutrition_model.pt'
# Model path
trained_model_path = r'model/malnutrition_model.pt'

# Model definition (must match the one used during training)
class MalnutritionModel(nn.Module):
    def __init__(self):
        super(MalnutritionModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=2, stride=2, padding=0)
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
data_trans = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor()
])

# Load model at startup
def load_model():
    model = MalnutritionModel()
    model.load_state_dict(torch.load(trained_model_path, map_location=device))
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
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        input_image = data_trans(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model_instance(input_image)
        _, predicted_class = torch.max(output.data, 1)
        class_label_map = {0: 'Malnourished', 1: 'Nourished'}
        predicted_label = class_label_map.get(predicted_class.item(), 'Unknown')
        return JSONResponse({
            "prediction": predicted_label,
            "api_call_count": api_call_count,
            "model_accuracy": model_accuracy
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/")
def root():
    return {"message": "Malnutrition Model API. Use /predict to POST an image and get Prediction."}
