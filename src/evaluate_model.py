import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.data import Dataset

# Define Paths Based on New Project Structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Root directory
MODEL_PATH = os.path.join(BASE_DIR, "models", "recognition", "final_model.h5")

# Use processed data for validation/testing
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")  # Contains images directly

# Load the trained model
print("[INFO] Loading trained model...")
model = load_model(MODEL_PATH)

# Image Preprocessing
IMG_SIZE = (224, 224)  # Must match the input size of your model

def load_and_preprocess_image(img_path):
    """Load and preprocess an image."""
    img = image.load_img(img_path, target_size=IMG_SIZE)  # Load image
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array

# Load images
print("[INFO] Loading processed images for evaluation...")
image_paths = [os.path.join(PROCESSED_DIR, fname) for fname in os.listdir(PROCESSED_DIR) if fname.endswith(('.png', '.jpg', '.jpeg'))]

if not image_paths:
    print(" No images found in processed folder! Check preprocessing step.")
    exit()

# Convert images to TensorFlow dataset
image_tensors = np.vstack([load_and_preprocess_image(img_path) for img_path in image_paths])

# Perform Predictions
print("[INFO] Running model evaluation on processed dataset...")
predictions = model.predict(image_tensors)

# Print Predictions
for i, img_path in enumerate(image_paths):
    predicted_label = np.argmax(predictions[i])  # Get class index
    print(f" {os.path.basename(img_path)} → Predicted Label: {predicted_label}")

print("\n Model Evaluation Completed Successfully!")