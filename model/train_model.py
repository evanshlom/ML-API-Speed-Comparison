### Trains intentionally slow random forest model

import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import pickle
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Generate data with MANY features
X, y = make_regression(n_samples=10000, n_features=100, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a MASSIVE Random Forest (these are notoriously slow in ONNX)
model = RandomForestRegressor(
    n_estimators=200,      # 200 trees!
    max_depth=20,          # Very deep trees
    min_samples_split=2,   # Allow very detailed splits
    min_samples_leaf=1,    # Maximum detail
    random_state=42,
    n_jobs=1              # Single thread to make it slower
)

print("Training ultra-slow Random Forest...")
print("200 trees, depth 20, 100 features - this will be VERY slow for inference!")

model.fit(X_train, y_train)

# Save original model
with open('ultra_slow_forest.pkl', 'wb') as f:
    pickle.dump(model, f)

# Convert to ONNX
initial_type = [('float_input', FloatTensorType([None, 100]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Save ONNX model
with open('linear_regression.onnx', 'wb') as f:  # Keep same name
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
    "input_shape": [100],
    "output_shape": [1],
    "model_type": "random_forest",
    "framework": "sklearn"
}

with open('model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print("Ultra-slow Random Forest trained!")
print(f"Train R2 score: {model.score(X_train, y_train):.4f}")
print(f"Test R2 score: {model.score(X_test, y_test):.4f}")
print("200 trees × 20 depth × 100 features = MAXIMUM SLOWNESS! 🐌")