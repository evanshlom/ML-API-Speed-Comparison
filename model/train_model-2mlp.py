### Trains deep learning regression model

import numpy as np
import json
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Generate more complex data
X, y = make_regression(n_samples=10000, n_features=20, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a much larger neural network
model = MLPRegressor(
    hidden_layer_sizes=(100, 50, 25),  # 3 hidden layers
    max_iter=1000,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Save original model
with open('neural_network.pkl', 'wb') as f:
    pickle.dump((model, scaler), f)

# Convert to ONNX
initial_type = [('float_input', FloatTensorType([None, 20]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Save ONNX model
with open('linear_regression.onnx', 'wb') as f:  # Keep same name for compatibility
    f.write(onnx_model.SerializeToString())

# Create test data with 20 features
test_samples = X_test_scaled[:10].tolist()
expected_outputs = model.predict(X_test_scaled[:10]).tolist()

test_data = {
    "samples": test_samples,
    "expected": expected_outputs
}

with open('test_data.json', 'w') as f:
    json.dump(test_data, f, indent=2)

# Model info
model_info = {
    "input_shape": [20],
    "output_shape": [1],
    "model_type": "neural_network",
    "framework": "sklearn"
}

with open('model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print("Neural network trained and exported successfully!")
print(f"Train R2 score: {model.score(X_train_scaled, y_train):.4f}")
print(f"Test R2 score: {model.score(X_test_scaled, y_test):.4f}")
print("Model has 3 hidden layers with 100, 50, and 25 neurons")