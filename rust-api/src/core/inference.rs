use anyhow::Result;
use ort::{Environment, Session, SessionBuilder, Value};
use serde_json;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use ndarray::{ArrayD, CowArray};

static INFERENCE_ENGINE: OnceLock<InferenceEngine> = OnceLock::new();

pub struct InferenceEngine {
    session: Session,
    model_info: HashMap<String, serde_json::Value>,
}

impl InferenceEngine {
    pub fn new() -> Result<Self> {
        let environment = Arc::new(Environment::builder().build()?);
        let session = SessionBuilder::new(&environment)?
            .with_model_from_file("../model/linear_regression.onnx")?;
        
        let model_info_str = std::fs::read_to_string("../model/model_info.json")?;
        let model_info: HashMap<String, serde_json::Value> = serde_json::from_str(&model_info_str)?;
        
        Ok(Self { session, model_info })
    }
    
    pub fn predict_single(&self, features: &[f32]) -> Result<f32> {
        let array = ArrayD::from_shape_vec(vec![1, features.len()], features.to_vec())?;
        let cow_array = CowArray::from(array);
        let input_tensor = Value::from_array(self.session.allocator(), &cow_array)?;
        let outputs = self.session.run(vec![input_tensor])?;
        
        let output_tensor = outputs[0].try_extract::<f32>()?;
        let view = output_tensor.view();
        Ok(view[[0, 0]])
    }
    
    pub fn predict_batch(&self, features: &[Vec<f32>]) -> Result<Vec<f32>> {
        let batch_size = features.len();
        let feature_size = features[0].len();
        
        let mut flat_features = Vec::with_capacity(batch_size * feature_size);
        for row in features {
            flat_features.extend_from_slice(row);
        }
        
        let array = ArrayD::from_shape_vec(vec![batch_size, feature_size], flat_features)?;
        let cow_array = CowArray::from(array);
        let input_tensor = Value::from_array(self.session.allocator(), &cow_array)?;
        let outputs = self.session.run(vec![input_tensor])?;
        
        let output_tensor = outputs[0].try_extract::<f32>()?;
        let view = output_tensor.view();
        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            results.push(view[[i, 0]]);
        }
        Ok(results)
    }
    
    pub fn get_model_info(&self) -> &HashMap<String, serde_json::Value> {
        &self.model_info
    }
}

pub fn get_inference_engine() -> Result<&'static InferenceEngine> {
    INFERENCE_ENGINE.get_or_init(|| InferenceEngine::new().unwrap());
    Ok(INFERENCE_ENGINE.get().unwrap())
}