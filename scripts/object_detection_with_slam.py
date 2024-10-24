  GNU nano 4.8                                                                                                                                                         object_detection_with_slam.py                                                                                                                                                                    #!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Path to the saved model
#model_path = '/home/aiot/myproject/src/object_detection_2/scripts/models/new_model_path.h5'
#model = tf.keras.models.load_model(model_path)
model_path = '/home/aiot/myproject/src/object_detection/models/new_model_path.keras'
model = tf.keras.models.load_model(model_path)

# Load the model
model = load_model(model_path)
print("Model loaded successfully.")

# Data preparation
def preprocess_data(raw_data):
    return np.array(raw_data).reshape(-1, 5, 1)

# Example input data
raw_input_data = np.random.random((1, 5, 1))
processed_input_data = preprocess_data(raw_input_data)

# Make predictions
predictions = model.predict(processed_input_data)
print("Predictions:")
print(predictions)

# Example test data and labels
test_data = np.random.random((10, 5, 1))
test_labels = np.random.randint(0, 5, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=5)

# Evaluate the model
results = model.evaluate(test_data, test_labels)
print("Evaluation results:")
print(f"Loss: {results[0]}, Accuracy: {results[1]}")

# Example SLAM integration
def integrate_slam_predictions(slam_data, model_predictions):
    combined_results = np.concatenate((slam_data, model_predictions), axis=-1)
    return combined_results

# Example SLAM data
slam_data = np.random.random((1, 10))
integrated_results = integrate_slam_predictions(slam_data, predictions)
print("Integrated Results:")
print(integrated_results)
