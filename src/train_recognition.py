import os
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ---------------------------
# Configurations and Paths
# ---------------------------
DATA_PROCESSED_DIR = os.path.join("data", "processed")
ANNOTATIONS_FILE = os.path.join("data", "annotations", "annotations.csv")
MODEL_SAVE_DIR = os.path.join("models", "recognition")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Model Hyperparameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4

# ---------------------------
# Step 1: Prepare the DataFrame
# ---------------------------
df = pd.read_csv(ANNOTATIONS_FILE)
df["class"] = df["crop"].str.lower() + "_" + df["disease"].str.lower()

df = df.dropna(subset=["class"])
df = df[df["class"] != "unknown_unknown"]

train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df["class"])

# Define number of classes
num_classes = train_df["class"].nunique()
print(f"Number of classes: {num_classes}")

# ---------------------------
# Step 3: Set Up Image Data Generators
# ---------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=DATA_PROCESSED_DIR,
    x_col="filename",
    y_col="class",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

validation_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=DATA_PROCESSED_DIR,
    x_col="filename",
    y_col="class",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ---------------------------
# Step 4: Build the Model with MobileNetV2
# ---------------------------
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=IMAGE_SIZE + (3,))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
predictions = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

# ---------------------------
# Step 5: Set Up Callbacks for Training
# ---------------------------
checkpoint_path = os.path.join(MODEL_SAVE_DIR, "best_model.h5")
checkpoint = ModelCheckpoint(checkpoint_path, monitor="val_accuracy", verbose=1,
                             save_best_only=True, mode="max")

early_stop = EarlyStopping(monitor="val_loss", patience=5, verbose=1, restore_best_weights=True)

# ---------------------------
# Step 6: Train the Model
# ---------------------------
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stop]
)

# Save final model
final_model_path = os.path.join(MODEL_SAVE_DIR, "final_model.h5")
model.save(final_model_path)
print(f"Final model saved to: {final_model_path}")
