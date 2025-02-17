import os
import numpy as np
import tensorflow as tf
from PIL import Image

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "recognition")
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, "model.tflite")

# Define image properties
IMG_SIZE = (224, 224)

# Define the class names (must match training configuration)
# Replace these with your actual class names as used during training.
CLASS_NAMES = sorted([
    "potato_early", "potato_healthy", "potato_late",
    "tomato_bacterial", "tomato_blight", "tomato_early",
    "tomato_healthy", "tomato_late", "tomato_leaf", "tomato_septoria"
])
print("[INFO] Model classes:", CLASS_NAMES)

def preprocess_image(image_path):
    """
    Load an image from the specified path, resize it to IMG_SIZE,
    and preprocess it to match the input requirements of MobileNetV2.
    MobileNetV2 expects inputs in the range [-1, 1].
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    # Normalize: scale pixel values from [0, 255] to [-1, 1]
    img_array = (img_array / 127.5) - 1.0
    # Expand dimensions to add batch size dimension (1, 224, 224, 3)
    return np.expand_dims(img_array, axis=0)

# Load the TFLite model and allocate tensors
print("[INFO] Loading TFLite model from:", TFLITE_MODEL_PATH)
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define a sample image path (update with an actual image path from your processed folder)
SAMPLE_IMAGE_PATH = os.path.join(BASE_DIR, "data", "processed", "Pepper__bell_Bacterial_spot_19.jpg")

# Preprocess the sample image
img_input = preprocess_image(SAMPLE_IMAGE_PATH)

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], img_input)

# Run inference
print("[INFO] Running inference...")
interpreter.invoke()

# Retrieve the prediction results
predictions = interpreter.get_tensor(output_details[0]['index'])
predicted_index = np.argmax(predictions[0])
confidence = float(np.max(predictions[0]))
predicted_class = CLASS_NAMES[predicted_index] if predicted_index < len(CLASS_NAMES) else "Unknown"

print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {confidence:.2f}")
