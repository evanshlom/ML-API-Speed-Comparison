from pydantic import BaseModel, ConfigDict
from typing import List

class PredictRequest(BaseModel):
    features: List[float]

class BatchPredictRequest(BaseModel):
    features: List[List[float]]

class PredictResponse(BaseModel):
    prediction: float

class BatchPredictResponse(BaseModel):
    predictions: List[float]

class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    status: str
    model_loaded: bool

class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    input_shape: List[int]
    output_shape: List[int]
    model_type: str
    framework: str