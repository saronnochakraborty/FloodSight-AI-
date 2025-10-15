import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# =========================================================
# CONFIGURATION
# =========================================================
CSV_PATH = "C:\\Users\\Desktop\\Desktop\\New folder (2)\\tensorflow_model\\assets\\dataset.csv"
SAVE_DIR = "C:\\Users\\Desktop\\Desktop\\New folder (2)\\tensorflow_model\\saved_model"
os.makedirs(SAVE_DIR, exist_ok=True)

# Feature columns (based on your augmented dataset)
FEATURE_COLS = [
    'temperature_c', 'humidity_pct', 'upstream_flow', 'soil_moisture',
    'previous_day_rainfall', 'wind_speed', 'pressure', 'evaporation_rate',
    'catchment_area', 'river_level', 'elevation', 'distance_to_river',
    'land_use_index', 'solar_radiation'
]

TARGETS = ['rainfall_mm', 'flood_risk_pct']

# =========================================================
# LOAD DATA
# =========================================================
print("📂 Loading dataset...")
df = pd.read_csv(CSV_PATH)

X = df[FEATURE_COLS].values
y = df[TARGETS].values

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

joblib.dump(scaler_X, os.path.join(SAVE_DIR, "scaler_X.joblib"))
joblib.dump(scaler_y, os.path.join(SAVE_DIR, "scaler_y.joblib"))
print("✅ Scalers saved.")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.15, random_state=42)

# =========================================================
# MODEL ARCHITECTURE
# =========================================================
print("🧠 Building model...")

inputs = Input(shape=(len(FEATURE_COLS),))

x = Dense(128, activation='relu')(inputs)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

rain_output = Dense(1, name='rainfall_output')(x)
flood_output = Dense(1, name='flood_output')(x)

model = Model(inputs=inputs, outputs=[rain_output, flood_output])
model.compile(
    optimizer=Adam(1e-3),
    loss='mse',
    metrics={'rainfall_output': 'mae', 'flood_output': 'mae'}
)

# =========================================================
# TRAINING
# =========================================================
print("🚀 Training started...")

callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]

history = model.fit(
    X_train, [y_train[:, 0], y_train[:, 1]],
    validation_data=(X_test, [y_test[:, 0], y_test[:, 1]]),
    epochs=100,
    batch_size=256,
    callbacks=callbacks,
    verbose=1
)

print("✅ Training complete.")

# =========================================================
# SAVE MODELS
# =========================================================
keras_path = os.path.join(SAVE_DIR, "final_model.keras")
tflite_path = os.path.join(SAVE_DIR, "model.tflite")

model.save(keras_path)
print(f"✅ Keras model saved at: {keras_path}")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"✅ TFLite model saved at: {tflite_path}")
print("🎉 Training pipeline complete!")
