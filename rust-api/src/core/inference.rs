use anyhow::Result;
use ort::{Environment, Session, SessionBuilder, Value};
use serde_json;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use ndarray::{ArrayD, CowArray};
use std::fs;

static INFERENCE_ENGINE: OnceLock<InferenceEngine> = OnceLock::new();

pub struct InferenceEngine {
    session: Session,
    model_info: HashMap<String, serde_json::Value>,
}

impl InferenceEngine {
    pub fn new() -> Result<Self> {
        let environment = Arc::new(Environment::builder().build()?);
        
        // Try different model paths
        let possible_paths = [
            "/app/model/linear_regression.onnx",
            "model/linear_regression.onnx",
            "../model/linear_regression.onnx",
            "linear_regression.onnx"
        ];
        
        let mut model_path = None;
        for path in &possible_paths {
            if fs::metadata(path).is_ok() {
                model_path = Some(path);
                println!("Found ONNX model at: {}", path);
                break;
            }
        }
        
        let model_path = model_path.ok_or_else(|| {
            anyhow::anyhow!("Could not find ONNX model. Tried: {:?}", possible_paths)
        })?;
        
        let session = SessionBuilder::new(&environment)?
            .with_model_from_file(model_path)?;
        
        // Try different paths for model info
        let info_paths = [
            "/app/model/model_info.json",
            "model/model_info.json",
            "../model/model_info.json"
        ];
        
        let mut model_info = HashMap::new();
        for info_path in &info_paths {
            if let Ok(model_info_str) = fs::read_to_string(info_path) {
                if let Ok(info) = serde_json::from_str(&model_info_str) {
                    model_info = info;
                    println!("Found model info at: {}", info_path);
                    break;
                }
            }
        }
        
        // Default if not found
        if model_info.is_empty() {
            model_info.insert("input_shape".to_string(), serde_json::json!([5]));
            model_info.insert("output_shape".to_string(), serde_json::json!([1]));
            model_info.insert("model_type".to_string(), serde_json::json!("linear_regression"));
            model_info.insert("framework".to_string(), serde_json::json!("sklearn"));
            println!("Using default model info");
        }
        
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
    INFERENCE_ENGINE.get_or_init(|| {
        match InferenceEngine::new() {
            Ok(engine) => {
                println!("Inference engine created successfully!");
                engine
            }
            Err(e) => {
                println!("Failed to create inference engine: {}", e);
                panic!("Could not initialize inference engine: {}", e);
            }
        }
    });
    Ok(INFERENCE_ENGINE.get().unwrap())
}