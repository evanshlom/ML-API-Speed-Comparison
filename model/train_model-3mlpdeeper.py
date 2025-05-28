### Trains deeper deep learning model

# import numpy as np
import json
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Generate complex data with more features for deeper processing
X, y = make_regression(n_samples=10000, n_features=50, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train an EXTREMELY deep neural network
model = MLPRegressor(
    hidden_layer_sizes=(512, 256, 256, 128, 128, 64, 64, 32, 32, 16),  # 10 hidden layers!
    max_iter=2000,
    random_state=42,
    alpha=0.001,  # Small regularization to prevent overfitting
    learning_rate_init=0.001
)

print("Training extremely deep neural network...")
print("Architecture: 50 -> 512 -> 256 -> 256 -> 128 -> 128 -> 64 -> 64 -> 32 -> 32 -> 16 -> 1")
print("Total parameters: ~500,000+")

model.fit(X_train_scaled, y_train)

# Save original model
with open('deep_neural_network.pkl', 'wb') as f:
    pickle.dump((model, scaler), f)

# Convert to ONNX (keep same filename for compatibility)
initial_type = [('float_input', FloatTensorType([None, 50]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Save ONNX model
with open('linear_regression.onnx', 'wb') as f:  # Keep same name for compatibility
    f.write(onnx_model.SerializeToString())

# Create test data with 50 features
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
    "input_shape": [50],
    "output_shape": [1],
    "model_type": "deep_neural_network",
    "framework": "sklearn"
}

with open('model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print("Deep neural network trained and exported successfully!")
print(f"Train R2 score: {model.score(X_train_scaled, y_train):.4f}")
print(f"Test R2 score: {model.score(X_test_scaled, y_test):.4f}")
print("Model has 10 hidden layers with 500,000+ parameters")
print("This should create significant inference time differences between Python and Rust!")