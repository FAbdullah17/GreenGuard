import os
import tensorflow as tf

# Define paths based on the project structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "recognition")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "final_model.h5")
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, "model.tflite")

print("[INFO] Loading the trained Keras model from:", FINAL_MODEL_PATH)
model = tf.keras.models.load_model(FINAL_MODEL_PATH)

print("[INFO] Converting the Keras model to TensorFlow Lite format...")
# Create the TFLiteConverter from the Keras model
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable optimizations for size and latency (optional but recommended for mobile deployment)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model to TFLite format
tflite_model = converter.convert()

# Save the TFLite model to file
with open(TFLITE_MODEL_PATH, "wb") as f:
    f.write(tflite_model)

print(f"[INFO] TensorFlow Lite model saved to: {TFLITE_MODEL_PATH}")
