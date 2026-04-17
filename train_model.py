"""
Face Mask Detection - Model Training
Uses MobileNetV2 as base (transfer learning) + custom classifier head
Dataset: With Mask / Without Mask classes
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

# ─── Config ───────────────────────────────────────────────────────────────────
DATASET_DIR   = "dataset"          # expects dataset/with_mask/ & dataset/without_mask/
MODEL_SAVE    = "models/mask_detector.h5"
PLOT_SAVE     = "models/training_plot.png"

INIT_LR       = 1e-4
EPOCHS        = 20
BATCH_SIZE    = 32
IMG_SIZE      = (224, 224)
CLASSES       = ["with_mask", "without_mask"]

# ─── Load & Preprocess Dataset ────────────────────────────────────────────────
print("[INFO] Loading images from dataset...")

data, labels = [], []
for label in CLASSES:
    path = os.path.join(DATASET_DIR, label)
    for img_file in os.listdir(path):
        img_path = os.path.join(path, img_file)
        try:
            img   = load_img(img_path, target_size=IMG_SIZE)
            arr   = img_to_array(img)
            arr   = preprocess_input(arr)
            data.append(arr)
            labels.append(label)
        except Exception as e:
            print(f"  Skipping {img_path}: {e}")

data   = np.array(data,   dtype="float32")
labels = np.array(labels)

lb      = LabelBinarizer()
labels  = lb.fit_transform(labels)
labels  = tf.keras.utils.to_categorical(labels, 2)

(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.20, stratify=labels, random_state=42
)

print(f"[INFO] Training samples : {len(trainX)}")
print(f"[INFO] Testing  samples : {len(testX)}")

# ─── Data Augmentation ────────────────────────────────────────────────────────
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# ─── Build Model ──────────────────────────────────────────────────────────────
print("[INFO] Building model (MobileNetV2 + custom head)...")

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(224, 224, 3))
)
base_model.trainable = False          # freeze base initially

head = base_model.output
head = AveragePooling2D(pool_size=(7, 7))(head)
head = Flatten()(head)
head = Dense(128, activation="relu")(head)
head = Dropout(0.5)(head)
head = Dense(2, activation="softmax")(head)

model = Model(inputs=base_model.input, outputs=head)

# ─── Compile ──────────────────────────────────────────────────────────────────
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(
    loss="binary_crossentropy",
    optimizer=opt,
    metrics=["accuracy"]
)
model.summary()

# ─── Callbacks ────────────────────────────────────────────────────────────────
callbacks = [
    ModelCheckpoint(
        MODEL_SAVE, monitor="val_accuracy", save_best_only=True, verbose=1
    ),
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
]

# ─── Train ────────────────────────────────────────────────────────────────────
print("[INFO] Training head layers...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
    steps_per_epoch=len(trainX) // BATCH_SIZE,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ─── Evaluate ─────────────────────────────────────────────────────────────────
print("[INFO] Evaluating model...")
predY       = model.predict(testX, batch_size=BATCH_SIZE)
predLabels  = np.argmax(predY, axis=1)
trueLabels  = np.argmax(testY, axis=1)

print(classification_report(trueLabels, predLabels, target_names=lb.classes_))

# Confusion matrix
cm = confusion_matrix(trueLabels, predLabels)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=lb.classes_, yticklabels=lb.classes_)
plt.title("Confusion Matrix"); plt.ylabel("True"); plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png")
print("[INFO] Confusion matrix saved to models/confusion_matrix.png")

# Training curves
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(H.history["loss"],     label="Train Loss")
plt.plot(H.history["val_loss"], label="Val Loss")
plt.title("Loss"); plt.xlabel("Epoch"); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(H.history["accuracy"],     label="Train Acc")
plt.plot(H.history["val_accuracy"], label="Val Acc")
plt.title("Accuracy"); plt.xlabel("Epoch"); plt.legend()

plt.tight_layout()
plt.savefig(PLOT_SAVE)
print(f"[INFO] Training plot saved to {PLOT_SAVE}")
print(f"[INFO] Model saved to {MODEL_SAVE}")
