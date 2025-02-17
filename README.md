# GreenGuard: AI-Powered Plant Disease Detection

GreenGuard is a **mobile-centric** project that uses **computer vision** and **deep learning** to detect plant diseases from images captured by a smartphone camera. The system is robust, efficient, and accurate, aiming to help farmers and researchers identify diseases quickly and take timely action. Below is a detailed breakdown of the entire workflow, from planning to deployment.

---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Dataset and Preprocessing](#dataset-and-preprocessing)  
3. [Project Phases](#project-phases)  
   - [Phase 1: Planning & Requirements](#phase-1-planning--requirements)  
   - [Phase 2: Data Collection & Annotation](#phase-2-data-collection--annotation)  
   - [Phase 3: Model Development & Training](#phase-3-model-development--training)  
   - [Model Evaluation](#model-evaluation)  
   - [TensorFlow Lite Conversion](#tensorflow-lite-conversion)  
   - [Flutter Integration](#flutter-integration)  
4. [Project Structure](#project-structure)  
5. [Installation & Usage](#installation--usage)  
6. [Future Enhancements](#future-enhancements)  
7. [License](#license)

---

## Project Overview
GreenGuard leverages a **deep learning model** (MobileNetV2) to classify plant images into various **crop-disease** categories. It is designed for **mobile use**, ensuring **low-latency inference** and **on-device** processing using **TensorFlow Lite**.

### Key Objectives
- **Accurate Disease Detection:** Achieve high classification accuracy using preprocessed images.  
- **Efficient On-Device Inference:** Deploy a lightweight TFLite model for real-time predictions on smartphones.  
- **User-Friendly Mobile App:** Provide an intuitive Flutter interface that allows farmers to capture images and receive instant feedback.

---

## Dataset and Preprocessing
**Dataset:** [PlantVillage](https://data.mendeley.com/datasets/tywbtsjrjv/1)  
- We curated a subset of **10–15 crops** with multiple diseases each.  
- Images were **resized** to `224×224` pixels and **normalized** for consistency.  
- Annotations were stored in `data/annotations/annotations.csv`.

**Preprocessing Steps:**
1. **Curating & Cleaning:** Filenames standardized (`<crop>_<disease>_<id>.jpg`).  
2. **Resizing & Denoising:** Using OpenCV’s `fastNlMeansDenoisingColored`.  
3. **Annotations:** Created a combined label (`crop_disease`) to simplify classification.

---

## Project Phases

### **Phase 1: Planning & Requirements**
1. **Define Objectives & Success Criteria:**  
   - Target accuracy ≥ 90%, inference time < 1 second on a mid-range smartphone.  
2. **Identify Crops & Diseases:**  
   - Chose relevant crops from PlantVillage and major diseases for each.  
3. **Hardware & Software Selection:**  
   - MobileNetV2 for the model.  
   - Flutter for the front-end.  
   - TFLite for on-device inference.  

### **Phase 2: Data Collection & Annotation**
1. **Data Gathering & Filtration:**  
   - Raw images in `data/raw/`.  
   - Removed poor-quality images or unknown labels.  
2. **Preprocessing Pipeline:**  
   - `preprocessing.py` script resizes, denoises, and saves images to `data/processed/`.  
3. **Annotation Generation:**  
   - `annotations.csv` created, combining `crop` and `disease` as the final class.

### **Phase 3: Model Development & Training**
1. **Model Selection:**  
   - **MobileNetV2** for a lightweight, efficient solution.  
2. **Training Script (`train_recognition.py`):**  
   - Splits data into train/validation (85/15).  
   - Uses `ImageDataGenerator` for augmentation.  
   - Compiles and trains the model, saving `final_model.h5`.  

#### Model Evaluation
- **`evaluate_model.py`** loads `final_model.h5`, runs validation or test sets, and prints accuracy/loss.  
- Ensures the model meets performance goals (accuracy, speed).

#### TensorFlow Lite Conversion
- **`convert_to_tflite.py`** converts `final_model.h5` to `model.tflite`, enabling mobile deployment.  
- Optional optimization (quantization) can further reduce size and improve speed.

#### Flutter Integration
- The `mob_app/` folder contains a Flutter project.  
- `model.tflite` is placed in `assets/`.  
- Using the `tflite_flutter` package, the mobile app loads the model, preprocesses the captured image, and performs inference in real time.

---

## Project Structure
```
GreenGuard/
├── data/
│   ├── raw/                   # Raw images
│   ├── processed/             # Preprocessed images
│   └── annotations/           # Annotations (CSV, JSON)
├── models/
│   └── recognition/
│       ├── final_model.h5     # Trained model
│       ├── model.tflite       # TFLite model for mobile
├── src/
│   ├── preprocessing.py       # Data preprocessing script
│   ├── train_recognition.py   # Model training script
│   ├── evaluate_model.py      # Model evaluation
│   ├── convert_to_tflite.py   # TFLite conversion
│   └── app.py                 # (Optional) Flask API or server integration
├── mob_app/
│   ├── pubspec.yaml           # Flutter dependencies
│   ├── android/lib/
│   │   ├── main.dart          # Main entry point for Flutter
│   │   ├── home_page.dart     # UI for capturing and inferring images
│   └── assets/
│       ├── model.tflite       # TFLite model
│       └── sample_image.jpg   # Sample image
├── venv/
├── requirements.txt
└── README.md
```

---

## Installation & Usage

### 1. **Clone the Repository**
```bash
git clone https://github.com/FAbdullah17/GreenGuard.git
cd GreenGuard
```

### 2. **Set Up Python Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate   # On Windows
pip install -r requirements.txt
```

### 3. **Data Preprocessing**
```bash
python src/preprocessing.py
```

### 4. **Model Training**
```bash
python src/train_recognition.py
```

### 5. **Evaluate Model**
```bash
python src/evaluate_model.py
```

### 6. **Convert to TFLite (Optional)**
```bash
python src/convert_to_tflite.py
```

### 7. **Flutter App Setup**
```bash
cd mob_app
flutter pub get
flutter run
```

---

## Future Enhancements
- Real-Time Camera Integration  
- Fine-Tuning & Unfreezing Layers  
- GPU Delegates & Quantization  
- Backend/Cloud Integration  
- Multi-Language Support  

---

## License
This project is licensed under the [MIT License](LICENSE). Feel free to modify and distribute as needed.

---

**Enjoy using GreenGuard!** If you have questions, please open an issue or contact the maintainers.

