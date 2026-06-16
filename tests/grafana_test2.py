"""
Grafana Dashboard Data Generator - Short Test (40 seconds)
Generates diverse traffic patterns for quick Grafana dashboard testing
"""

import requests
import time
import random
import base64
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# ============================================================================
# CONFIGURATION
# ============================================================================

API_URL = "https://localhost"
VERIFY_SSL = False

# Complete list of 54 supported locations
ALL_LOCATIONS = [
    "Adelaide", "Albany", "Albury", "AliceSprings", "BadgerysCreek",
    "Badgingarra", "Balladonia", "Ballarat", "Bendigo", "Bridgetown",
    "Brisbane", "Broome", "Bunbury", "Cairns", "Canberra", "Cobar",
    "CoffsHarbour", "Dartmoor", "Darwin", "Devonport", "Esperance",
    "Geraldton", "GoldCoast", "Hobart", "Kalgoorlie", "Launceston",
    "Meekatharra", "Melbourne", "MelbourneAirport", "Mildura", "Moree",
    "MountGambier", "MountGinini", "Newcastle", "Nhil", "NorahHead",
    "NorfolkIsland", "Nuriootpa", "PearceRAAF", "Perth", "PerthAirport",
    "Portmacquarie", "Sale", "SalmonGums", "SunshineCoast", "Sydney",
    "SydneyAirport", "Townsville", "Tuggeranong", "Uluru", "WaggaWagga",
    "Walpole", "Watsonia", "Woomera"
]

# Different test scenarios (weighted for 40-second test)
TEST_SCENARIOS = [
    {"name": "predict_single", "weight": 35},      # Single predictions
    {"name": "predict_batch", "weight": 20},       # Batch predictions
    {"name": "locations", "weight": 15},           # Locations endpoint
    {"name": "health", "weight": 10},              # Health checks
    {"name": "admin_endpoints", "weight": 10},     # Admin endpoints
    {"name": "unauthenticated", "weight": 10}      # Without auth (should fail)
]

def get_auth_header(username="andrey", password="andrey"):
    """Generates Basic Auth header."""
    credentials = f"{username}:{password}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return {"Authorization": f"Basic {encoded}", "Content-Type": "application/json"}

def get_random_locations(count=5):
    """Returns random locations from the complete list."""
    return random.sample(ALL_LOCATIONS, min(count, len(ALL_LOCATIONS)))

def generate_random_payload(locations):
    """Generates a random but realistic prediction payload."""
    location = random.choice(locations)
    
    humidity = random.uniform(20, 95)
    rain_today = random.choice(["Yes", "No"])
    pressure = random.uniform(990, 1030)
    rainfall = random.uniform(0, 20) if rain_today == "Yes" else random.uniform(0, 2)
    wind_speed = random.uniform(10, 80)
    humidity_9am = random.uniform(max(0, humidity - 20), min(100, humidity + 10))
    
    return {
        "location": location,
        "humidity_3pm": round(humidity, 1),
        "rain_today": rain_today,
        "wind_gust_speed": round(wind_speed, 1),
        "rainfall": round(rainfall, 1),
        "pressure_3pm": round(pressure, 1),
        "humidity_9am": round(humidity_9am, 1)
    }

def send_single_prediction(locations):
    """Sends a single prediction request."""
    payload = generate_random_payload(locations)
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            headers=get_auth_header(),
            verify=VERIFY_SSL,
            timeout=10
        )
        end_time = time.time()
        return {
            "success": response.status_code == 200,
            "status": response.status_code,
            "time_ms": (end_time - start_time) * 1000,
            "endpoint": "/predict",
            "method": "POST"
        }
    except Exception as e:
        return {"success": False, "status": 0, "time_ms": 0, "endpoint": "/predict", "method": "POST", "error": str(e)}

def send_batch_prediction(locations):
    """Sends a batch prediction request."""
    num_samples = random.randint(2, 4)
    samples = [generate_random_payload(locations) for _ in range(num_samples)]
    payload = {"samples": samples}
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_URL}/predict/batch",
            json=payload,
            headers=get_auth_header(),
            verify=VERIFY_SSL,
            timeout=10
        )
        end_time = time.time()
        return {
            "success": response.status_code == 200,
            "status": response.status_code,
            "time_ms": (end_time - start_time) * 1000,
            "endpoint": "/predict/batch",
            "method": "POST"
        }
    except Exception as e:
        return {"success": False, "status": 0, "time_ms": 0, "endpoint": "/predict/batch", "method": "POST", "error": str(e)}

def send_locations_request():
    """Sends a locations request."""
    try:
        start_time = time.time()
        response = requests.get(
            f"{API_URL}/locations",
            headers=get_auth_header(),
            verify=VERIFY_SSL,
            timeout=10
        )
        end_time = time.time()
        return {
            "success": response.status_code == 200,
            "status": response.status_code,
            "time_ms": (end_time - start_time) * 1000,
            "endpoint": "/locations",
            "method": "GET"
        }
    except Exception as e:
        return {"success": False, "status": 0, "time_ms": 0, "endpoint": "/locations", "method": "GET", "error": str(e)}

def send_health_request():
    """Sends a health check request (no auth needed)."""
    try:
        start_time = time.time()
        response = requests.get(
            f"{API_URL}/health",
            verify=VERIFY_SSL,
            timeout=10
        )
        end_time = time.time()
        return {
            "success": response.status_code == 200,
            "status": response.status_code,
            "time_ms": (end_time - start_time) * 1000,
            "endpoint": "/health",
            "method": "GET"
        }
    except Exception as e:
        return {"success": False, "status": 0, "time_ms": 0, "endpoint": "/health", "method": "GET", "error": str(e)}

def send_admin_request():
    """Sends an admin endpoint request."""
    endpoint = random.choice(["/model/info", "/model/features", "/metrics"])
    try:
        start_time = time.time()
        response = requests.get(
            f"{API_URL}{endpoint}",
            headers=get_auth_header(),
            verify=VERIFY_SSL,
            timeout=10
        )
        end_time = time.time()
        return {
            "success": response.status_code == 200,
            "status": response.status_code,
            "time_ms": (end_time - start_time) * 1000,
            "endpoint": endpoint,
            "method": "GET"
        }
    except Exception as e:
        return {"success": False, "status": 0, "time_ms": 0, "endpoint": endpoint, "method": "GET", "error": str(e)}

def send_unauthenticated_request(locations):
    """Sends an unauthenticated request (should fail with 401)."""
    endpoint = random.choice(["/predict", "/predict/batch", "/model/info"])
    try:
        start_time = time.time()
        
        if endpoint == "/predict":
            payload = generate_random_payload(locations)
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                verify=VERIFY_SSL,
                timeout=10
            )
        elif endpoint == "/predict/batch":
            samples = [generate_random_payload(locations) for _ in range(2)]
            payload = {"samples": samples}
            response = requests.post(
                f"{API_URL}/predict/batch",
                json=payload,
                verify=VERIFY_SSL,
                timeout=10
            )
        else:
            response = requests.get(
                f"{API_URL}/model/info",
                verify=VERIFY_SSL,
                timeout=10
            )
        
        end_time = time.time()
        return {
            "success": response.status_code == 401,  # Unauthorized is expected
            "status": response.status_code,
            "time_ms": (end_time - start_time) * 1000,
            "endpoint": endpoint,
            "method": "GET" if endpoint == "/model/info" else "POST"
        }
    except Exception as e:
        return {"success": False, "status": 0, "time_ms": 0, "endpoint": endpoint, "method": "POST", "error": str(e)}

def print_header(title):
    """Prints a formatted header."""
    print("\n" + "="*70)
    print(f"📊 {title}")
    print("="*70)

def run_short_test():
    """
    Runs a short 40-second test with diverse traffic patterns.
    """
    # Select 8 random cities for this test
    locations = get_random_locations(8)
    
    print_header("GRAFANA SHORT TEST (40 seconds)")
    print(f"📍 API URL: {API_URL}")
    print(f"⏱️  Duration: 40 seconds")
    print(f"🏙️  Locations for this run: {', '.join(locations)}")
    print("="*70)
    
    total_requests = 0
    results = []
    successful = 0
    failed = 0
    endpoint_stats = {}
    
    start_time = time.time()
    
    print("\n🔄 Generating requests for 40 seconds...")
    print("   Press Ctrl+C to stop early\n")
    
    # Define the request functions
    request_functions = [
        (send_single_prediction, "predict_single", locations),
        (send_batch_prediction, "predict_batch", locations),
        (send_locations_request, "locations", None),
        (send_health_request, "health", None),
        (send_admin_request, "admin_endpoints", None),
        (send_unauthenticated_request, "unauthenticated", locations)
    ]
    
    # Weighted selection for 40-second test
    weights = [35, 20, 15, 10, 10, 10]  # Sum = 100
    
    try:
        while time.time() - start_time < 40:
            # Select a random scenario based on weights
            selected = random.choices(request_functions, weights=weights, k=1)[0]
            func, name, loc = selected
            
            # Execute the request
            if loc is not None:
                result = func(loc)
            else:
                result = func()
            
            results.append(result)
            total_requests += 1
            
            # Track endpoint statistics
            endpoint_key = f"{result.get('method', 'UNKNOWN')} {result.get('endpoint', 'UNKNOWN')}"
            if endpoint_key not in endpoint_stats:
                endpoint_stats[endpoint_key] = {"total": 0, "success": 0}
            endpoint_stats[endpoint_key]["total"] += 1
            
            if result["success"]:
                successful += 1
                endpoint_stats[endpoint_key]["success"] += 1
            else:
                failed += 1
            
            # Progress indicator every 10 requests
            if total_requests % 10 == 0:
                print(f"   ✓ {total_requests} requests sent ({successful} successful)")
            
            # Small delay to avoid overwhelming the API
            time.sleep(0.05)  # ~20 requests per second max
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped by user")
    
    elapsed = time.time() - start_time
    
    # Print summary
    print_header("TEST SUMMARY")
    print(f"⏱️  Duration: {elapsed:.1f} seconds")
    print(f"📨 Total requests: {len(results)}")
    print(f"✅ Successful: {successful} ({successful/len(results)*100:.1f}%)")
    print(f"❌ Failed: {failed} ({failed/len(results)*100:.1f}%)")
    
    print(f"\n📊 Endpoint Statistics:")
    for endpoint, stats in sorted(endpoint_stats.items(), key=lambda x: x[1]["total"], reverse=True):
        success_rate = (stats["success"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"   {endpoint}: {stats['total']} requests ({success_rate:.1f}% success)")
    
    # Calculate throughput
    requests_per_sec = len(results) / elapsed
    print(f"\n🚀 Throughput: {requests_per_sec:.2f} req/s")
    
    # Quick health check
    print_header("FINAL STATUS")
    
    # Check if we got any successful responses
    if successful > 0:
        print("✅ API is responding to authenticated requests")
    else:
        print("❌ No successful requests - check API status")
    
    if any(r["status"] == 401 for r in results):
        print("✅ Authentication is working (401 responses seen)")
    else:
        print("ℹ️  No 401 responses seen")
    
    return results

def run_burst_test(num_requests=30):
    """Runs a quick burst test."""
    locations = get_random_locations(5)
    
    print_header("BURST TEST (30 requests)")
    print(f"🚀 Sending {num_requests} requests in parallel...")
    
    # Mix of endpoints
    def create_task():
        scenario = random.choice(TEST_SCENARIOS)
        if scenario["name"] == "predict_single":
            return send_single_prediction(locations)
        elif scenario["name"] == "predict_batch":
            return send_batch_prediction(locations)
        elif scenario["name"] == "locations":
            return send_locations_request()
        elif scenario["name"] == "health":
            return send_health_request()
        elif scenario["name"] == "admin_endpoints":
            return send_admin_request()
        else:
            return send_unauthenticated_request(locations)
    
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [executor.submit(create_task) for _ in range(num_requests)]
        results = [f.result() for f in futures]
    elapsed = time.time() - start_time
    
    successful = sum(1 for r in results if r["success"])
    status_codes = {}
    for r in results:
        status_codes[r["status"]] = status_codes.get(r["status"], 0) + 1
    
    print(f"\n📊 Burst Results:")
    print(f"   Duration: {elapsed:.2f} seconds")
    print(f"   Successful: {successful}/{num_requests} ({successful/num_requests*100:.1f}%)")
    print(f"   Status codes: {status_codes}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Short 40-second test for Grafana dashboards")
    parser.add_argument("-b", "--burst", type=int, default=0, help="Run burst test with N requests instead")
    
    args = parser.parse_args()
    
    print("="*70)
    print("🧪 GRAFANA SHORT TEST (40 seconds)")
    print("="*70)
    print()
    print("This tool runs a quick 40-second test with diverse traffic patterns.")
    print("It tests: single predictions, batch predictions, locations, health, admin endpoints")
    print()
    
    if args.burst > 0:
        run_burst_test(args.burst)
    else:
        run_short_test()
    
    print("\n✅ Done! Check your Grafana dashboard for updated metrics.")
    print("   Dashboard URL: http://34.247.195.225:3000")
    print()
    print("💡 Tip: Run the longer test for more data:")
    print("   python3 tests/grafana_alternative.py -d 5 -r 3")