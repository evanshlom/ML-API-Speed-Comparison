import requests
import time
import json
import statistics
import concurrent.futures
from typing import List, Dict

class APIBenchmark:
    def __init__(self, python_url: str = "http://localhost:8000", rust_url: str = "http://localhost:8001"):
        self.python_url = python_url
        self.rust_url = rust_url
        
        # Load test data
        with open('../model/test_data.json', 'r') as f:
            self.test_data = json.load(f)
    
    def single_request_benchmark(self, url: str, num_requests: int = 100) -> Dict:
        """Benchmark single prediction requests"""
        latencies = []
        
        for i in range(num_requests):
            features = self.test_data['samples'][i % len(self.test_data['samples'])]
            payload = {"features": features}
            
            start_time = time.perf_counter()
            response = requests.post(f"{url}/predict", json=payload)
            end_time = time.perf_counter()
            
            if response.status_code == 200:
                latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        return {
            "avg_latency_ms": statistics.mean(latencies),
            "p50_latency_ms": statistics.median(latencies),
            "p95_latency_ms": statistics.quantiles(latencies, n=20)[18],  # 95th percentile
            "p99_latency_ms": statistics.quantiles(latencies, n=100)[98],  # 99th percentile
            "total_requests": len(latencies)
        }
    
    def batch_request_benchmark(self, url: str, batch_sizes: List[int] = [1, 10, 50, 100]) -> Dict:
        """Benchmark batch prediction requests"""
        results = {}
        
        for batch_size in batch_sizes:
            features = self.test_data['samples'][:batch_size]
            payload = {"features": features}
            
            # Warm up
            requests.post(f"{url}/predict/batch", json=payload)
            
            latencies = []
            for _ in range(10):  # 10 iterations per batch size
                start_time = time.perf_counter()
                response = requests.post(f"{url}/predict/batch", json=payload)
                end_time = time.perf_counter()
                
                if response.status_code == 200:
                    latencies.append((end_time - start_time) * 1000)
            
            results[f"batch_{batch_size}"] = {
                "avg_latency_ms": statistics.mean(latencies),
                "throughput_per_sec": batch_size / (statistics.mean(latencies) / 1000)
            }
        
        return results
    
    def concurrent_benchmark(self, url: str, concurrent_users: int = 10, requests_per_user: int = 10) -> Dict:
        """Benchmark concurrent requests"""
        def make_requests(user_id: int):
            latencies = []
            for i in range(requests_per_user):
                features = self.test_data['samples'][i % len(self.test_data['samples'])]
                payload = {"features": features}
                
                start_time = time.perf_counter()
                response = requests.post(f"{url}/predict", json=payload)
                end_time = time.perf_counter()
                
                if response.status_code == 200:
                    latencies.append((end_time - start_time) * 1000)
            return latencies
        
        start_time = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(make_requests, i) for i in range(concurrent_users)]
            all_latencies = []
            for future in concurrent.futures.as_completed(futures):
                all_latencies.extend(future.result())
        end_time = time.perf_counter()
        
        total_requests = len(all_latencies)
        total_time = end_time - start_time
        
        return {
            "avg_latency_ms": statistics.mean(all_latencies),
            "total_requests": total_requests,
            "total_time_sec": total_time,
            "requests_per_sec": total_requests / total_time,
            "concurrent_users": concurrent_users
        }
    
    def run_full_benchmark(self):
        """Run complete benchmark suite"""
        print("Starting benchmarks...")
        
        results = {
            "python": {},
            "rust": {},
            "timestamp": time.time()
        }
        
        # Test both APIs
        for name, url in [("python", self.python_url), ("rust", self.rust_url)]:
            print(f"\nBenchmarking {name.upper()} API...")
            
            try:
                # Single request benchmark
                print("  Running single request benchmark...")
                results[name]["single_requests"] = self.single_request_benchmark(url)
                
                # Batch request benchmark
                print("  Running batch request benchmark...")
                results[name]["batch_requests"] = self.batch_request_benchmark(url)
                
                # Concurrent benchmark
                print("  Running concurrent request benchmark...")
                results[name]["concurrent_requests"] = self.concurrent_benchmark(url)
                
            except Exception as e:
                print(f"  Error benchmarking {name}: {e}")
                results[name]["error"] = str(e)
        
        # Save results
        with open('results/benchmark_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        self.print_comparison(results)
        return results
    
    def print_comparison(self, results: Dict):
        """Print benchmark comparison"""
        print("\n" + "="*60)
        print("BENCHMARK RESULTS COMPARISON")
        print("="*60)
        
        if "error" not in results["python"] and "error" not in results["rust"]:
            # Single request comparison
            py_single = results["python"]["single_requests"]["avg_latency_ms"]
            rust_single = results["rust"]["single_requests"]["avg_latency_ms"]
            speedup = py_single / rust_single
            
            print(f"\nSingle Request Latency:")
            print(f"  Python: {py_single:.2f}ms")
            print(f"  Rust:   {rust_single:.2f}ms")
            print(f"  Speedup: {speedup:.2f}x")
            
            # Concurrent throughput comparison
            py_concurrent = results["python"]["concurrent_requests"]["requests_per_sec"]
            rust_concurrent = results["rust"]["concurrent_requests"]["requests_per_sec"]
            throughput_speedup = rust_concurrent / py_concurrent
            
            print(f"\nConcurrent Throughput:")
            print(f"  Python: {py_concurrent:.1f} req/sec")
            print(f"  Rust:   {rust_concurrent:.1f} req/sec")
            print(f"  Speedup: {throughput_speedup:.2f}x")

if __name__ == "__main__":
    benchmark = APIBenchmark()
    benchmark.run_full_benchmark()