import onnxruntime as ort
import numpy as np
import json
from typing import List

class ONNXInferenceEngine:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Load model info
        with open('../model/model_info.json', 'r') as f:
            self.model_info = json.load(f)
    
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

# Global inference engine
inference_engine = ONNXInferenceEngine("../model/linear_regression.onnx")