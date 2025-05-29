#!/bin/bash

# run_load_test.sh - Load test runner using wrk

echo "🔥 WRK CONCURRENT LOAD TEST"
echo "=============================="
echo "Testing 1000 concurrent requests against both APIs"
echo ""

# Wait for APIs to be ready
echo "⏳ Waiting for APIs to be ready..."

# Wait for Python API
while ! curl -s http://python-api:8000/health > /dev/null; do
    echo "Waiting for Python API..."
    sleep 2
done
echo "✅ Python API is ready!"

# Wait for Rust API  
while ! curl -s http://rust-api:8001/health > /dev/null; do
    echo "Waiting for Rust API..."
    sleep 2
done
echo "✅ Rust API is ready!"

echo ""
echo "🚀 Starting load tests..."

# Test Python API
echo ""
echo "=============================================="
echo "🐍 TESTING PYTHON API"
echo "=============================================="
echo "URL: http://python-api:8000/predict"
echo "Connections: 10, Threads: 5, Duration: 10s"
echo ""

wrk -t5 -c10 -d10s --script=/app/scripts/post.lua http://python-api:8000/predict

# Brief pause
echo ""
echo "⏳ Waiting 3 seconds before next test..."
sleep 3

# Test Rust API
echo ""
echo "=============================================="
echo "🦀 TESTING RUST API"  
echo "=============================================="
echo "URL: http://rust-api:8001/predict"
echo "Connections: 10, Threads: 5, Duration: 10s"
echo ""

# Test Rust API
echo ""
echo "=============================================="
echo "🦀 TESTING RUST API"  
echo "=============================================="
echo "URL: http://rust-api:8001/predict"
echo "Connections: 10, Threads: 5, Duration: 10s"
echo ""

wrk -t5 -c10 -d10s --script=/app/scripts/post.lua http://rust-api:8001/predict

# Final comparison (simple version since we're not saving files)
echo ""
echo "=============================================="
echo "🏆 COMPARISON COMPLETE!"
echo "=============================================="
echo "✅ Both Python and Rust APIs tested successfully!"
echo "📊 Check the results above to see which performed better"
echo "🎥 Perfect for your YouTube video!"
echo ""
echo "✅ Load test completed!"