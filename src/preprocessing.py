import os
import cv2
import glob
import pandas as pd

# Define directory paths
RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
ANNOTATIONS_DIR = os.path.join("data", "annotations")
ANNOTATIONS_FILE = os.path.join(ANNOTATIONS_DIR, "annotations.csv")

# Create the necessary directories if they do not exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

# Define the target image size (width, height)
TARGET_SIZE = (224, 224)

# Initialize a list to store annotation records
annotations = []

# Gather all images from RAW_DIR (supporting common extensions)
image_extensions = ["*.jpg", "*.jpeg", "*.png"]
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(RAW_DIR, ext)))

print(f"Found {len(image_files)} images in '{RAW_DIR}'.")

# Process each image one by one
for image_path in image_files:
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Unable to read image: {image_path}")
            continue

        # Resize the image to the target dimensions
        resized_image = cv2.resize(image, TARGET_SIZE)

        # Apply denoising to remove unwanted noise
        # h and hColor parameters can be tuned for best performance
        denoised_image = cv2.fastNlMeansDenoisingColored(
            resized_image, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21
        )

        # Save the processed image to PROCESSED_DIR with the same filename
        filename = os.path.basename(image_path)
        processed_path = os.path.join(PROCESSED_DIR, filename)
        cv2.imwrite(processed_path, denoised_image)

        # Extract annotations from the filename.
        # Assumes filename format: crop_disease_id.jpg (e.g., tomato_blight_001.jpg)
        base_name, _ = os.path.splitext(filename)
        parts = base_name.split("_")
        if len(parts) >= 2:
            crop = parts[0]
            disease = parts[1]
        else:
            crop = "unknown"
            disease = "unknown"

        # Append the annotation record to our list
        annotations.append({
            "filename": filename,
            "crop": crop,
            "disease": disease
        })

        print(f"Processed: {filename} | Crop: {crop} | Disease: {disease}")

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")

# After processing all images, create a CSV file for annotations
if annotations:
    df = pd.DataFrame(annotations)
    df.to_csv(ANNOTATIONS_FILE, index=False)
    print(f"\nAnnotations saved to '{ANNOTATIONS_FILE}'.")
else:
    print("No annotations were generated.")

print("Data preprocessing and annotation completed successfully.")