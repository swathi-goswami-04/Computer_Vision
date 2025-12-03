import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# --- 1. Configuration and Model Structure Definition ---
IMAGE_SIZE = 224
NUM_CLASSES = 1
MODEL_SAVE_PATH = 'final_best_model.pt'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Must redefine the model structure exactly as it was trained
def get_model_structure(num_classes):
    """Initializes the MobileNetV2 structure with the correct classification head."""
    # Load MobileNetV2 structure without weights
    model = models.mobilenet_v2(weights=None)
    
    # Replace the final fully connected layer (the "head")
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(num_ftrs, num_classes),
        nn.Sigmoid() 
    )
    # Set to evaluation mode immediately
    model.eval()
    return model

# --- 2. Preprocessing Function for Single Image ---
def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Applies necessary transforms (grayscale, resize, normalize) to a raw PIL image."""
    
    inference_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3), 
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    # Apply transforms and add batch dimension (C x H x W -> 1 x C x H x W)
    image_tensor = inference_transforms(image).unsqueeze(0) 
    
    return image_tensor

# --- 3. Core Inference Handler ---
@torch.no_grad() # Disable gradient calculation for faster inference
def predict_image(image: Image.Image, model: nn.Module) -> tuple[str, float]:
    """Runs inference on a single PIL image."""
    
    model.to(DEVICE)
    input_tensor = preprocess_image(image).to(DEVICE)
    
    # Run prediction
    output = model(input_tensor)
    probability = output.item()
    
    # Determine the class (0: Defect, 1: OK)
    if probability < 0.5:
        # Prediction is closer to 0 (Defect)
        prediction_label = "Defective"
        confidence_score = 1.0 - probability # Confidence in 'Defective'
    else:
        # Prediction is closer to 1 (OK)
        prediction_label = "OK"
        confidence_score = probability # Confidence in 'OK'

    return prediction_label, confidence_score

# --- 4. Model Initialization (To be run once when module is imported) ---
# This block attempts to load the model immediately upon import.

try:
    INFERENCE_MODEL = get_model_structure(NUM_CLASSES)
    INFERENCE_MODEL.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    print(f"Inference Model loaded successfully from {MODEL_SAVE_PATH}")
except FileNotFoundError:
    print(f"ERROR: Model file {MODEL_SAVE_PATH} not found. Gradio will show an error until trained model is saved.")
    INFERENCE_MODEL = None