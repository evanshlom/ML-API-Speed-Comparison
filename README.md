# ML API Speed Comparison: Rust vs Python

Compare ONNX inference performance between Rust (Axum) and Python (FastAPI).

## Quick Start

```bash
# 1. Train and export model
cd model && python train_model.py

# 2. Start both APIs
docker-compose up

# 3. Run benchmarks
cd benchmarks && python benchmark.py
```

## APIs
- Python: http://localhost:8000
- Rust: http://localhost:8001

## Results
Check `benchmarks/results/` for performance comparisons.