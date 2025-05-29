import requests
import time
import json
import statistics
import concurrent.futures
from typing import List, Dict

class ConcurrentLoadTester:
    def __init__(self, python_url: str = "http://python-api:8000", rust_url: str = "http://rust-api:8001"):
        self.python_url = python_url
        self.rust_url = rust_url
        
        # Load test data
        with open('/app/model/test_data.json', 'r') as f:
            self.test_data = json.load(f)
    
    def wait_for_apis(self):
        """Wait for both APIs to be ready"""
        print("Waiting for APIs to be ready...")
        for name, url in [("Python", self.python_url), ("Rust", self.rust_url)]:
            while True:
                try:
                    response = requests.get(f"{url}/health", timeout=5)
                    if response.status_code == 200:
                        print(f"{name} API is ready!")
                        break
                except:
                    pass
                time.sleep(2)
    
    def concurrent_load_test(self, url: str, total_requests: int = 1000, concurrent_workers: int = 50) -> Dict:
        """Run 1000 concurrent requests load test"""
        def make_request(request_id: int):
            features = self.test_data['samples'][request_id % len(self.test_data['samples'])]
            payload = {"features": features}
            
            start_time = time.perf_counter()
            try:
                response = requests.post(f"{url}/predict", json=payload, timeout=30)
                end_time = time.perf_counter()
                
                if response.status_code == 200:
                    return {
                        "success": True,
                        "latency_ms": (end_time - start_time) * 1000,
                        "request_id": request_id
                    }
                else:
                    return {
                        "success": False,
                        "error": f"HTTP {response.status_code}",
                        "request_id": request_id
                    }
            except Exception as e:
                end_time = time.perf_counter()
                return {
                    "success": False,
                    "error": str(e),
                    "latency_ms": (end_time - start_time) * 1000,
                    "request_id": request_id
                }
        
        print(f"Running {total_requests} concurrent requests with {concurrent_workers} workers...")
        
        start_time = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
            futures = [executor.submit(make_request, i) for i in range(total_requests)]
            
            results = []
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                completed += 1
                
                # Show progress every 100 requests
                if completed % 100 == 0:
                    print(f"  Completed {completed}/{total_requests} requests")
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Analyze results
        successful_results = [r for r in results if r.get('success', False)]
        failed_results = [r for r in results if not r.get('success', False)]
        
        if successful_results:
            latencies = [r['latency_ms'] for r in successful_results]
            
            return {
                "total_requests": total_requests,
                "successful_requests": len(successful_results),
                "failed_requests": len(failed_results),
                "success_rate_percent": (len(successful_results) / total_requests) * 100,
                "total_time_sec": total_time,
                "requests_per_sec": len(successful_results) / total_time,
                "avg_latency_ms": statistics.mean(latencies),
                "median_latency_ms": statistics.median(latencies),
                "p95_latency_ms": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies),
                "p99_latency_ms": statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max(latencies),
                "min_latency_ms": min(latencies),
                "max_latency_ms": max(latencies)
            }
        else:
            return {
                "total_requests": total_requests,
                "successful_requests": 0,
                "failed_requests": len(failed_results),
                "success_rate_percent": 0,
                "total_time_sec": total_time,
                "error": "All requests failed"
            }
    
    def run_concurrent_load_test(self):
        """Run concurrent load test on both APIs"""
        print("🔥 CONCURRENT LOAD TEST - 1000 REQUESTS")
        print("="*50)
        
        # Wait for APIs to be ready
        self.wait_for_apis()
        
        results = {
            "python": {},
            "rust": {},
            "timestamp": time.time(),
            "test_config": {
                "total_requests": 1000,
                "concurrent_workers": 50
            }
        }
        
        # Test both APIs
        for name, url in [("python", self.python_url), ("rust", self.rust_url)]:
            print(f"\n🚀 Testing {name.upper()} API...")
            
            try:
                results[name] = self.concurrent_load_test(url, total_requests=1000, concurrent_workers=50)
                print(f"✅ {name.upper()} test completed!")
                
            except Exception as e:
                print(f"❌ Error testing {name}: {e}")
                results[name]["error"] = str(e)
        
        # Save results
        with open('/app/results/concurrent_load_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        self.print_comparison(results)
        return results
    
    def print_comparison(self, results: Dict):
        """Print load test comparison"""
        print("\n" + "="*60)
        print("🏆 CONCURRENT LOAD TEST RESULTS")
        print("="*60)
        
        if "error" not in results["python"] and "error" not in results["rust"]:
            # Throughput comparison
            py_rps = results["python"]["requests_per_sec"]
            rust_rps = results["rust"]["requests_per_sec"]
            throughput_speedup = rust_rps / py_rps
            
            print(f"\n📊 Throughput (Requests/Second):")
            print(f"  Python: {py_rps:.1f} req/sec")
            print(f"  Rust:   {rust_rps:.1f} req/sec")
            print(f"  🚀 Rust is {throughput_speedup:.2f}x faster!")
            
            # Total time comparison
            py_time = results["python"]["total_time_sec"]
            rust_time = results["rust"]["total_time_sec"]
            time_speedup = py_time / rust_time
            
            print(f"\n⏱️  Total Time:")
            print(f"  Python: {py_time:.2f} seconds")
            print(f"  Rust:   {rust_time:.2f} seconds")
            print(f"  ⚡ Rust finished {time_speedup:.2f}x faster!")
            
            # Latency comparison (P95)
            py_p95 = results["python"]["p95_latency_ms"]
            rust_p95 = results["rust"]["p95_latency_ms"]
            latency_speedup = py_p95 / rust_p95
            
            print(f"\n📈 P95 Latency:")
            print(f"  Python: {py_p95:.1f}ms")
            print(f"  Rust:   {rust_p95:.1f}ms")
            print(f"  ⚡ Rust is {latency_speedup:.2f}x faster!")
            
            # Success rates
            py_success = results["python"]["success_rate_percent"]
            rust_success = results["rust"]["success_rate_percent"]
            
            print(f"\n✅ Success Rate:")
            print(f"  Python: {py_success:.1f}%")
            print(f"  Rust:   {rust_success:.1f}%")
            
        else:
            if "error" in results["python"]:
                print(f"❌ Python API failed: {results['python']['error']}")
            if "error" in results["rust"]:
                print(f"❌ Rust API failed: {results['rust']['error']}")

if __name__ == "__main__":
    tester = ConcurrentLoadTester()
    tester.run_concurrent_load_test()