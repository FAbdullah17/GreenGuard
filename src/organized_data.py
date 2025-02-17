import os
import glob
import shutil

# Define dataset paths
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
ANNOTATIONS_DIR = "data/annotations"

# Ensure directories exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

# Get all subdirectories (each one corresponds to a crop and disease)
subdirs = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))]

# Counter for renaming
image_id = 1

for subdir in subdirs:
    subdir_path = os.path.join(RAW_DIR, subdir)
    
    # Extract crop and disease name from folder name (assuming format: Crop___Disease)
    if "___" in subdir:
        crop, disease = subdir.split("___")
    else:
        crop, disease = subdir, "unknown"

    # Get all images inside the subdirectory
    image_files = glob.glob(os.path.join(subdir_path, "*.jpg"))

    for image_file in image_files:
        # Create a new standardized filename
        new_filename = f"{crop}_{disease}_{image_id}.jpg"
        new_path = os.path.join(RAW_DIR, new_filename)

        # Move and rename image
        shutil.move(image_file, new_path)
        image_id += 1

    # Remove the now-empty subdirectory
    os.rmdir(subdir_path)

print("Dataset reorganization completed successfully.")