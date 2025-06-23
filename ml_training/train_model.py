# train_model.py
print("🚀 Starting MindMig training script...")

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Add
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# === Paths ===
DATA_PATH = "dataset/migraine_dataset_500 (1).csv"  # ✅ FIXED PATH
MODEL_DIR = "../models"                             # ✅ Save outside ml_training/
MODEL_PATH = os.path.join(MODEL_DIR, "mindmig_resnet_model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
COLUMNS_PATH = os.path.join(MODEL_DIR, "columns.pkl")

# === Create model directory if not exists ===
os.makedirs(MODEL_DIR, exist_ok=True)

# === Load dataset ===
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# === Encode categorical ===
df["Gender"] = df["Gender"].map({'Male': 1, 'Female': 0})
df["Physical Activity"] = df["Physical Activity"].map({
    'None': 0, '1–2 days/week': 1, '3–5 days/week': 2, 'Daily': 3
})
df["Skipped Meals"] = df["Skipped Meals"].map({'Yes': 1, 'No': 0})
df["Menstruating"] = df["Menstruating"].map({'No': 0, 'Yes': 1, 'Not applicable': 2})
df["Migraine"] = df["Migraine"].map({'Yes': 1, 'No': 0})

# === Features and target ===
X = df.drop(columns=["Migraine"])
y = df["Migraine"]
X_encoded = pd.get_dummies(X).fillna(0)

# Save column order for prediction use
with open(COLUMNS_PATH, "wb") as f:
    pickle.dump(list(X_encoded.columns), f)

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, stratify=y, random_state=42
)

# === Balance classes with SMOTE ===
smote = SMOTE(sampling_strategy=0.8, random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# === Add small Gaussian noise ===
noise_factor = 0.01
X_train += noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)

# === Scale features ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

# === Reshape for CNN input ===
X_train_cnn = np.expand_dims(X_train_scaled.astype("float32"), axis=-1)
X_test_cnn = np.expand_dims(X_test_scaled.astype("float32"), axis=-1)
y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

# === ResNet block ===
def resnet_block(x, filters, kernel_size):
    skip = x
    x = Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
    x = Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
    skip = Conv1D(filters, 1, padding='same')(skip)
    return Add()([x, skip])

# === Build ResNet model ===
def build_resnet_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, kernel_size=2, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = resnet_block(x, 64, 2)
    x = MaxPooling1D(pool_size=2)(x)
    x = resnet_block(x, 32, 2)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, x)
    model.compile(optimizer=Adam(0.0005), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# === Train model ===
print("🚀 Starting training...")
model = build_resnet_model(input_shape=(X_train_cnn.shape[1], 1))
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train_cnn, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# === Evaluate model ===
loss, acc = model.evaluate(X_test_cnn, y_test)
print(f"\n✅ ResNet Test Accuracy: {acc:.4f}")

# === Save model ===
model.save(MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")

# === Evaluation results ===
y_pred = (model.predict(X_test_cnn) > 0.5).astype("int32")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# === Plot accuracy & loss ===
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.legend()

plt.tight_layout()
plt.show()


