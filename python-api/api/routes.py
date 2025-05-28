from fastapi import APIRouter, HTTPException
from api.models import (
    PredictRequest, PredictResponse,
    BatchPredictRequest, BatchPredictResponse,
    HealthResponse, ModelInfoResponse
)
from core.inference import inference_engine

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        prediction = inference_engine.predict_single(request.features)
        return PredictResponse(prediction=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    try:
        predictions = inference_engine.predict_batch(request.features)
        return BatchPredictResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", model_loaded=True)

@router.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    info = inference_engine.get_model_info()
    return ModelInfoResponse(**info)