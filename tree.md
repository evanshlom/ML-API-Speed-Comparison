# ML API Speed Comparison Project

```
ml-api-benchmark/
├── README.md
├── requirements.txt                 # Python dependencies
├── Cargo.toml                      # Rust workspace config
├── .gitignore
│
├── model/                          # Shared model artifacts
│   ├── train_model.py             # Script to train and export model
│   ├── linear_regression.pkl      # Original sklearn model
│   ├── linear_regression.onnx     # Exported ONNX model
│   ├── test_data.json            # Sample test data
│   └── model_info.json           # Model metadata (input/output shapes, etc.)
│
├── python-api/                    # FastAPI implementation
│   ├── main.py                   # FastAPI app entry point
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py            # API endpoints
│   │   └── models.py            # Pydantic models for request/response
│   ├── core/
│   │   ├── __init__.py
│   │   ├── inference.py         # ONNX inference logic
│   │   └── config.py            # Configuration settings
│   ├── requirements.txt         # Python API specific deps
│   └── Dockerfile              # Python API containerization
│
├── rust-api/                     # Rust implementation
│   ├── Cargo.toml              # Rust API dependencies
│   ├── src/
│   │   ├── main.rs             # Entry point
│   │   ├── lib.rs              # Library exports
│   │   ├── api/
│   │   │   ├── mod.rs
│   │   │   ├── handlers.rs     # Request handlers
│   │   │   └── models.rs       # Request/response structs
│   │   ├── core/
│   │   │   ├── mod.rs
│   │   │   ├── inference.rs    # ONNX inference logic
│   │   │   └── config.rs       # Configuration
│   │   └── utils/
│   │       ├── mod.rs
│   │       └── error.rs        # Error handling
│   └── Dockerfile             # Rust API containerization
│
├── benchmarks/                  # Performance testing
│   ├── benchmark.py            # Benchmark script
│   ├── load_test.py           # Load testing with multiple concurrent requests
│   ├── results/               # Benchmark results
│   │   ├── rust_results.json
│   │   ├── python_results.json
│   │   └── comparison.md
│   └── scripts/
│       ├── start_rust.sh      # Script to start Rust API
│       ├── start_python.sh    # Script to start Python API
│       └── run_benchmarks.sh  # Run full benchmark suite
│
├── docker-compose.yml          # For running both APIs together
└── docs/
    ├── api_spec.md            # API specification
    ├── setup.md               # Setup instructions
    └── results.md             # Performance comparison results
```

## Key Dependencies

### Python API (FastAPI)
```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
onnxruntime==1.16.3
pydantic==2.5.0
numpy==1.24.3
scikit-learn==1.3.0  # For model training
```

### Rust API (Cargo.toml)
```toml
[dependencies]
axum = "0.7"
tokio = { version = "1.0", features = ["full"] }
ort = "1.16"  # ONNX Runtime for Rust
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
tracing = "0.1"
tracing-subscriber = "0.3"
tower = "0.4"
tower-http = { version = "0.5", features = ["cors"] }
```

## API Endpoints (Both implementations)
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions
- `GET /health` - Health check
- `GET /model/info` - Model metadata