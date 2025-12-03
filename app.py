## app.py

import gradio as gr
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import cv2 
import os # For checking example files

# Import the necessary functions and data from the handler file
from inference_handler import predict_image, get_model_structure, MODEL_SAVE_PATH, DEVICE, preprocess_image, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES, IMAGE_SIZE

# --- 1. Load Model (Using the imported structure/path) ---
try:
    INFERENCE_MODEL = get_model_structure(NUM_CLASSES)
    INFERENCE_MODEL.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    INFERENCE_MODEL.to(DEVICE)
    INFERENCE_MODEL.eval()
    MODEL_LOADED = True
    print("Gradio app initialized: Model loaded.")
except FileNotFoundError:
    MODEL_LOADED = False
    print(f"ERROR: Model file {MODEL_SAVE_PATH} not found. Cannot run demo.")


# --- 2. Grad-CAM Implementation (Explainability) ---
def generate_grad_cam(model, image: Image.Image, target_layer_index=15):
    """Generates a Grad-CAM heatmap overlay to show model attention."""
    
    if not MODEL_LOADED:
        return image, "Model Not Loaded"

    # Set up hooks and input tensor
    target_layer = model.features[target_layer_index]
    features, gradients = [], []

    def save_features(module, input, output): features.append(output.cpu())
    def save_gradients(module, grad_input, grad_output): gradients.append(grad_output[0].cpu())

    hook_handle_f = target_layer.register_forward_hook(save_features)
    hook_handle_g = target_layer.register_full_backward_hook(save_gradients)

    input_tensor = preprocess_image(image).to(DEVICE)
    input_tensor.requires_grad_(True) 

    # Forward pass
    output = model(input_tensor)
    
    # Backward pass
    model.zero_grad()
    one_hot = torch.zeros_like(output)
    # Target the single Sigmoid output (index 0)
    one_hot[0][0] = 1.0 
    output.backward(gradient=one_hot.to(DEVICE), retain_graph=False)

    # Compute Grad-CAM
    pooled_gradients = torch.mean(gradients[-1], dim=[0, 2, 3])
    feature_map = features[-1][0]
    
    for i in range(pooled_gradients.shape[0]):
        feature_map[i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(feature_map, dim=0).relu().detach().numpy()

    # Post-processing and Overlay
    img_array = np.array(image.convert("RGB"))
    
    # Normalize and color map
    heatmap = np.uint8(255 * (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8))
    heatmap_colored = cv2.applyColorMap(cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0])), cv2.COLORMAP_JET)
    
    # Alpha blend the heatmap onto the image
    overlaid_img = cv2.addWeighted(img_array, 0.6, heatmap_colored, 0.4, 0)

    # Clean up hooks
    hook_handle_f.remove()
    hook_handle_g.remove()

    return Image.fromarray(overlaid_img)


# --- 3. Gradio Interface Function ---
def gradio_predict_wrapper(input_img: Image.Image):
    """Main function called by Gradio, runs prediction and CAM generation."""
    if not MODEL_LOADED:
        return input_img, f"❌ ERROR: Model file '{MODEL_SAVE_PATH}' not found. Check server logs."

    # 1. Run core prediction
    label, confidence = predict_image(input_img, INFERENCE_MODEL)
    
    # 2. Generate Grad-CAM visualization
    cam_img = generate_grad_cam(INFERENCE_MODEL, input_img)

    # 3. Format output
    output_text = f"**Prediction:** **{label}**\n"
    output_text += f"**Confidence:** {confidence:.2f}"
    
    return cam_img, output_text


# --- 4. Gradio Interface Definition ---

# Define the examples, ensuring the paths are correct relative to where you run the app.py
example_paths = [
    "./casting_product_data/test/def_front/cast_def_0_100.jpeg", 
    "./casting_product_data/test/ok_front/cast_ok_0_100.jpeg"
]
# Only use examples that exist
existing_examples = [p for p in example_paths if os.path.exists(p)]

interface = gr.Interface(
    fn=gradio_predict_wrapper,
    inputs=gr.Image(type="pil", label="Upload Casting Image for Inspection"),
    outputs=[
        gr.Image(type="pil", label="Defect Localization (Grad-CAM Heatmap)"),
        gr.Markdown(label="Inspection Result")
    ],
    title="🏭 Real-Time Manufacturing Quality Control (MobileNetV2)",
    description=(
        "Upload an image of a cast product (or use the example) to check for defects. "
        "The **Transfer Learning** model classifies the product, and the **Heatmap (Grad-CAM)** "
        "shows which areas the model focused on to make its prediction."
    ),
    examples=existing_examples,
    
)

# Launch the app
if __name__ == "__main__":
    interface.launch(inbrowser=True)