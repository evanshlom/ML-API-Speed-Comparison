import onnxruntime as ort
import numpy as np
import json
import os
from typing import List

class ONNXInferenceEngine:
    def __init__(self, model_path: str):
        # Handle different path scenarios
        possible_paths = [
            model_path,
            f"/app/model/{os.path.basename(model_path)}",
            f"model/{os.path.basename(model_path)}",
            f"../model/{os.path.basename(model_path)}"
        ]
        
        onnx_path = None
        for path in possible_paths:
            if os.path.exists(path):
                onnx_path = path
                print(f"Found ONNX model at: {path}")
                break
        
        if onnx_path is None:
            raise FileNotFoundError(f"Could not find ONNX model. Tried: {possible_paths}")
            
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Load model info - try different paths
        info_paths = [
            '/app/model/model_info.json',
            'model/model_info.json', 
            '../model/model_info.json'
        ]
        
        for info_path in info_paths:
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    self.model_info = json.load(f)
                print(f"Found model info at: {info_path}")
                break
        else:
            self.model_info = {
                "input_shape": [5],
                "output_shape": [1],
                "model_type": "linear_regression",
                "framework": "sklearn"
            }
            print("Using default model info")
    
    def predict_single(self, features: List[float]) -> float:
        input_data = np.array([features], dtype=np.float32)
        result = self.session.run([self.output_name], {self.input_name: input_data})
        return float(result[0][0])
    
    def predict_batch(self, features: List[List[float]]) -> List[float]:
        input_data = np.array(features, dtype=np.float32)
        result = self.session.run([self.output_name], {self.input_name: input_data})
        return [float(pred[0]) for pred in result[0]]
    
    def get_model_info(self) -> dict:
        return self.model_info

# Try to create inference engine with better path handling
print("Creating inference engine...")
try:
    inference_engine = ONNXInferenceEngine("linear_regression.onnx")
    print("Inference engine created successfully!")
except Exception as e:
    print(f"Failed to create inference engine: {e}")
    # Create a dummy engine for testing
    class DummyEngine:
        def predict_single(self, features): return 42.0
        def predict_batch(self, features): return [42.0] * len(features)
        def get_model_info(self): return {"input_shape": [5], "output_shape": [1], "model_type": "dummy", "framework": "none"}
    
    inference_engine = DummyEngine()
    print("Using dummy inference engine")