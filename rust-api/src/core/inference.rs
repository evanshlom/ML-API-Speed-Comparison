use anyhow::Result;
use ort::{Environment, Session, SessionBuilder, Value};
use serde_json;
use std::collections::HashMap;
use std::sync::OnceLock;

static INFERENCE_ENGINE: OnceLock<InferenceEngine> = OnceLock::new();

pub struct InferenceEngine {
    session: Session,
    model_info: HashMap<String, serde_json::Value>,
}

impl InferenceEngine {
    pub fn new() -> Result<Self> {
        let environment = Environment::builder().build()?;
        let session = SessionBuilder::new(&environment)?
            .with_model_from_file("../model/linear_regression.onnx")?;
        
        let model_info_str = std::fs::read_to_string("../model/model_info.json")?;
        let model_info: HashMap<String, serde_json::Value> = serde_json::from_str(&model_info_str)?;
        
        Ok(Self { session, model_info })
    }
    
    pub fn predict_single(&self, features: &[f32]) -> Result<f32> {
        let input_tensor = Value::from_array(self.session.allocator(), &[features])?;
        let outputs = self.session.run(vec![input_tensor])?;
        let output = outputs[0].extract_tensor::<f32>()?;
        Ok(output.view()[0])
    }
    
    pub fn predict_batch(&self, features: &[Vec<f32>]) -> Result<Vec<f32>> {
        let batch_size = features.len();
        let feature_size = features[0].len();
        
        let mut flat_features = Vec::with_capacity(batch_size * feature_size);
        for row in features {
            flat_features.extend_from_slice(row);
        }
        
        let input_tensor = Value::from_array(
            self.session.allocator(),
            &flat_features.into_iter()
                .collect::<Vec<f32>>()
                .chunks(feature_size)
                .collect::<Vec<_>>()
        )?;
        
        let outputs = self.session.run(vec![input_tensor])?;
        let output = outputs[0].extract_tensor::<f32>()?;
        Ok(output.view().iter().copied().collect())
    }
    
    pub fn get_model_info(&self) -> &HashMap<String, serde_json::Value> {
        &self.model_info
    }
}

pub fn get_inference_engine() -> Result<&'static InferenceEngine> {
    INFERENCE_ENGINE.get_or_try_init(|| InferenceEngine::new())
}