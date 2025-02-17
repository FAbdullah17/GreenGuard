import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "recognition", "final_model.h5")

print("[INFO] Loading trained model from:", MODEL_PATH)
model = load_model(MODEL_PATH)

IMG_SIZE = (224, 224)

CLASS_NAMES = sorted([
    "potato_early", "potato_healthy", "potato_late", 
    "tomato_bacterial", "tomato_blight", "tomato_early", 
    "tomato_healthy", "tomato_late", "tomato_leaf", "tomato_septoria"
])
print("[INFO] Model classes:", CLASS_NAMES)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Please attach an image with key 'image'."}), 400
    
    file = request.files["image"]
    
    try:
        img = image.load_img(file, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        predicted_class = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else "Unknown"
        
        return jsonify({"predicted_class": predicted_class, "confidence": confidence})
    
    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)