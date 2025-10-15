import tensorflow as tf
import os

def save_tf_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Force .keras extension
    if not path.endswith(".keras"):
        path += ".keras"
    model.save(path)
    print("Saved Keras model to", path)


def convert_and_save_tflite(keras_model_path, tflite_path):
    # Ensure correct extension
    if not keras_model_path.endswith(".keras"):
        keras_model_path += ".keras"
    model = tf.keras.models.load_model(keras_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print("Saved TFLite model to", tflite_path)

