use axum::{extract::Json, http::StatusCode, response::Json as ResponseJson};
use crate::api::models::*;
use crate::core::inference::get_inference_engine;

pub async fn predict(
    Json(payload): Json<PredictRequest>,
) -> Result<ResponseJson<PredictResponse>, StatusCode> {
    let engine = get_inference_engine().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    match engine.predict_single(&payload.features) {
        Ok(prediction) => Ok(ResponseJson(PredictResponse { prediction })),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

pub async fn predict_batch(
    Json(payload): Json<BatchPredictRequest>,
) -> Result<ResponseJson<BatchPredictResponse>, StatusCode> {
    let engine = get_inference_engine().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    match engine.predict_batch(&payload.features) {
        Ok(predictions) => Ok(ResponseJson(BatchPredictResponse { predictions })),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

pub async fn health() -> ResponseJson<HealthResponse> {
    ResponseJson(HealthResponse {
        status: "ok".to_string(),
        model_loaded: true,
    })
}

pub async fn model_info() -> Result<ResponseJson<ModelInfoResponse>, StatusCode> {
    let engine = get_inference_engine().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let info = engine.get_model_info();
    
    let input_shape: Vec<i32> = info["input_shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    
    let output_shape: Vec<i32> = info["output_shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    
    Ok(ResponseJson(ModelInfoResponse {
        input_shape,
        output_shape,
        model_type: info["model_type"].as_str().unwrap().to_string(),
        framework: info["framework"].as_str().unwrap().to_string(),
    }))
}