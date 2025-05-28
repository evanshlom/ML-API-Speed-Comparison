### Trains linear regression model

import numpy as np
import json
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import pickle
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Generate sample data
X, y = make_regression(n_samples=1000, n_features=5, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save original model
with open('linear_regression.pkl', 'wb') as f:
    pickle.dump(model, f)

# Convert to ONNX
initial_type = [('float_input', FloatTensorType([None, 5]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Save ONNX model
with open('linear_regression.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

# Create test data
test_samples = X_test[:10].tolist()
expected_outputs = model.predict(X_test[:10]).tolist()

test_data = {
    "samples": test_samples,
    "expected": expected_outputs
}

with open('test_data.json', 'w') as f:
    json.dump(test_data, f, indent=2)

# Model info
model_info = {
    "input_shape": [5],
    "output_shape": [1],
    "model_type": "linear_regression",
    "framework": "sklearn"
}

with open('model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print("Model trained and exported successfully!")
print(f"Train R2 score: {model.score(X_train, y_train):.4f}")
print(f"Test R2 score: {model.score(X_test, y_test):.4f}")