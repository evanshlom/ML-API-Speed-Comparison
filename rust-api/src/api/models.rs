use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub struct PredictRequest {
    pub features: Vec<f32>,
}

#[derive(Deserialize)]
pub struct BatchPredictRequest {
    pub features: Vec<Vec<f32>>,
}

#[derive(Serialize)]
pub struct PredictResponse {
    pub prediction: f32,
}

#[derive(Serialize)]
pub struct BatchPredictResponse {
    pub predictions: Vec<f32>,
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub model_loaded: bool,
}

#[derive(Serialize)]
pub struct ModelInfoResponse {
    pub input_shape: Vec<i32>,
    pub output_shape: Vec<i32>,
    pub model_type: String,
    pub framework: String,
}