import os
import numpy as np
import tensorflow as tf
import joblib

# =========================================================
# CONFIGURATION
# =========================================================

# Absolute paths relative to this file
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
MODEL_PATH = os.path.join(MODEL_DIR, "final_model.keras")
SCALER_X_PATH = os.path.join(MODEL_DIR, "scaler_X.joblib")
SCALER_Y_PATH = os.path.join(MODEL_DIR, "scaler_y.joblib")

# =========================================================
# LOAD MODEL AND SCALERS
# =========================================================
print("📂 Loading model and scalers...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"TensorFlow Keras model not found at {MODEL_PATH}")
if not os.path.exists(SCALER_X_PATH):
    raise FileNotFoundError(f"Scaler X file not found at {SCALER_X_PATH}")
if not os.path.exists(SCALER_Y_PATH):
    raise FileNotFoundError(f"Scaler Y file not found at {SCALER_Y_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
scaler_X = joblib.load(SCALER_X_PATH)
scaler_y = joblib.load(SCALER_Y_PATH)

print("✅ Model and scalers loaded successfully.")

# =========================================================
# DEFINE FEATURE ORDER (must match training)
# =========================================================
FEATURE_COLS = [
    'temperature_c', 'humidity_pct', 'upstream_flow', 'soil_moisture',
    'previous_day_rainfall', 'wind_speed', 'pressure', 'evaporation_rate',
    'catchment_area', 'river_level', 'elevation', 'distance_to_river',
    'land_use_index', 'solar_radiation'
]

# =========================================================
# PREDICTION FUNCTION
# =========================================================
def predict_weather(features_dict):
    """
    features_dict example:
    {
        'temperature_c': 29.5,
        'humidity_pct': 75,
        'upstream_flow': 120,
        'soil_moisture': 0.4,
        'previous_day_rainfall': 12.5,
        'wind_speed': 8,
        'pressure': 1008,
        'evaporation_rate': 3.1,
        'catchment_area': 245,
        'river_level': 2.8,
        'elevation': 120,
        'distance_to_river': 2.1,
        'land_use_index': 0.65,
        'solar_radiation': 310
    }
    """

    # Convert dict → numpy array
    input_values = np.array([[features_dict[feat] for feat in FEATURE_COLS]])

    # Scale input
    X_scaled = scaler_X.transform(input_values)

    # Predict with TensorFlow model
    rainfall_pred_scaled, flood_pred_scaled = model.predict(X_scaled)

    # Combine outputs for inverse transform
    y_scaled_combined = np.concatenate([rainfall_pred_scaled, flood_pred_scaled], axis=1)

    # Inverse scale
    y_pred = scaler_y.inverse_transform(y_scaled_combined)

    rainfall_pred, flood_pred = y_pred[0]

    return {
        "predicted_rainfall_mm": float(rainfall_pred),
        "predicted_flood_risk_pct": float(flood_pred)
    }

# =========================================================
# TEST RUN
# =========================================================
if __name__ == "__main__":
    sample = {
        'temperature_c': 28.5,
        'humidity_pct': 80,
        'upstream_flow': 110,
        'soil_moisture': 0.35,
        'previous_day_rainfall': 15.2,
        'wind_speed': 6.5,
        'pressure': 1007,
        'evaporation_rate': 3.0,
        'catchment_area': 210,
        'river_level': 3.2,
        'elevation': 130,
        'distance_to_river': 1.8,
        'land_use_index': 0.7,
        'solar_radiation': 295
    }

    preds = predict_weather(sample)
    print("🌦️ Predicted Rainfall (mm):", preds["predicted_rainfall_mm"])
    print("🌊 Predicted Flood Risk (%):", preds["predicted_flood_risk_pct"])
