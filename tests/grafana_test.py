"""
Grafana Dashboard Data Generator - Random Test Data
Generates random prediction metrics for testing Grafana dashboards
"""

import requests
import time
import random
import base64
import argparse
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# ============================================================================
# CONFIGURATION
# ============================================================================

API_URL = "https://localhost"
VERIFY_SSL = False
ADMIN_USERNAME = os.getenv("NGINX_ADMIN_USER")
ADMIN_PASSWORD = os.getenv("NGINX_ADMIN_PASSWORD")

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

# Weather scenarios
WEATHER_SCENARIOS = [
    {"name": "sunny", "humidity": (20, 40), "rain_today": "No", "pressure": (1015, 1025)},
    {"name": "cloudy", "humidity": (50, 70), "rain_today": "No", "pressure": (1010, 1020)},
    {"name": "rainy", "humidity": (75, 95), "rain_today": "Yes", "pressure": (998, 1010)},
    {"name": "storm", "humidity": (80, 98), "rain_today": "Yes", "pressure": (985, 1000)},
    {"name": "humid", "humidity": (70, 90), "rain_today": "No", "pressure": (1005, 1015)},
    {"name": "dry", "humidity": (10, 30), "rain_today": "No", "pressure": (1020, 1030)}
]

# Randomly select 10 cities for each test run (to avoid overload)
def get_random_locations():
    """Returns 10 random locations from the complete list."""
    return random.sample(ALL_LOCATIONS, 10)

def get_auth_header():
    """Generates Basic Auth header."""
    if not ADMIN_USERNAME or not ADMIN_PASSWORD:
        raise RuntimeError("Set NGINX_ADMIN_USER and NGINX_ADMIN_PASSWORD before running this Nginx/Grafana traffic script.")
    credentials = f"{ADMIN_USERNAME}:{ADMIN_PASSWORD}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return {"Authorization": f"Basic {encoded}", "Content-Type": "application/json"}

def generate_random_payload(locations):
    """Generates a random but realistic prediction payload."""
    location = random.choice(locations)
    scenario = random.choice(WEATHER_SCENARIOS)
    
    humidity = random.uniform(scenario["humidity"][0], scenario["humidity"][1])
    pressure = random.uniform(scenario["pressure"][0], scenario["pressure"][1])
    rainfall = random.uniform(0, 15) if scenario["rain_today"] == "Yes" else random.uniform(0, 2)
    wind_speed = random.uniform(10, 80)
    humidity_9am = random.uniform(humidity - 15, humidity + 5)
    humidity_9am = max(0, min(100, humidity_9am))
    
    return {
        "location": location,
        "humidity_3pm": round(humidity, 1),
        "rain_today": scenario["rain_today"],
        "wind_gust_speed": round(wind_speed, 1),
        "rainfall": round(rainfall, 1),
        "pressure_3pm": round(pressure, 1),
        "humidity_9am": round(humidity_9am, 1)
    }

def send_prediction(payload):
    """Sends a prediction request to the API."""
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
        
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "status": response.status_code,
                "time_ms": (end_time - start_time) * 1000,
                "prediction": data.get("prediction"),
                "rain_tomorrow": data.get("rain_tomorrow"),
                "confidence": data.get("confidence"),
                "location": payload["location"],
                "humidity": payload["humidity_3pm"],
                "rain_today": payload["rain_today"],
                "pressure": payload["pressure_3pm"]
            }
        else:
            return {
                "success": False,
                "status": response.status_code,
                "time_ms": (end_time - start_time) * 1000,
                "error": f"HTTP {response.status_code}"
            }
    except Exception as e:
        return {
            "success": False,
            "status": 0,
            "time_ms": 0,
            "error": str(e)
        }

def print_header(title):
    """Prints a formatted header."""
    print("\n" + "="*70)
    print(f"📊 {title}")
    print("="*70)

def run_data_generator(duration_minutes=5, requests_per_second=2):
    """
    Generates random prediction data for Grafana dashboards.
    
    Args:
        duration_minutes: How long to run the generator (minutes)
        requests_per_second: Number of requests per second
    """
    # Select random 10 cities for this test run
    locations = get_random_locations()
    
    print_header("GRAFANA DATA GENERATOR")
    print(f"📍 API URL: {API_URL}")
    print(f"⏱️  Duration: {duration_minutes} minutes")
    print(f"🚀 Rate: {requests_per_second} requests/second")
    print(f"🏙️  Total locations available: {len(ALL_LOCATIONS)}")
    print(f"🎲 Random locations for this run: {len(locations)}")
    print(f"   {', '.join(locations[:5])}... ({len(locations)-5} more)")
    print(f"🌦️  Weather scenarios: {len(WEATHER_SCENARIOS)}")
    print("="*70)
    
    total_requests = duration_minutes * 60 * requests_per_second
    interval = 1.0 / requests_per_second
    
    results = []
    successful = 0
    failed = 0
    predictions_by_location = {}
    confidences = []
    response_times = []
    
    start_time = time.time()
    
    print(f"\n🔄 Generating {total_requests} requests over {duration_minutes} minutes...")
    print("   Press Ctrl+C to stop early\n")
    
    try:
        for i in range(total_requests):
            # Generate random payload with selected locations
            payload = generate_random_payload(locations)
            
            # Send prediction
            result = send_prediction(payload)
            results.append(result)
            
            if result["success"]:
                successful += 1
                
                # Track predictions by location
                loc = result["location"]
                if loc not in predictions_by_location:
                    predictions_by_location[loc] = 0
                predictions_by_location[loc] += 1
                
                # Track confidence
                if result.get("confidence"):
                    confidences.append(result["confidence"])
                
                # Track response time
                response_times.append(result["time_ms"])
                
                # Progress indicator
                if (i + 1) % 50 == 0:
                    print(f"   ✓ {i+1}/{total_requests} requests sent ({successful} successful)")
            else:
                failed += 1
                if (i + 1) % 50 == 0:
                    print(f"   ⚠️  {i+1}/{total_requests} requests sent ({failed} failed)")
            
            # Wait to maintain rate
            if i < total_requests - 1:
                time.sleep(interval)
                
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped by user")
    
    elapsed = time.time() - start_time
    
    # Print summary
    print_header("GENERATION SUMMARY")
    print(f"⏱️  Duration: {elapsed:.1f} seconds")
    print(f"📨 Total requests: {len(results)}")
    print(f"✅ Successful: {successful} ({successful/len(results)*100:.1f}%)")
    print(f"❌ Failed: {failed} ({failed/len(results)*100:.1f}%)")
    
    if response_times:
        print(f"\n⏱️  Response Times:")
        print(f"   Min: {min(response_times):.2f} ms")
        print(f"   Max: {max(response_times):.2f} ms")
        print(f"   Avg: {sum(response_times)/len(response_times):.2f} ms")
    
    if confidences:
        print(f"\n🎯 Prediction Confidence:")
        print(f"   Min: {min(confidences)*100:.1f}%")
        print(f"   Max: {max(confidences)*100:.1f}%")
        print(f"   Avg: {sum(confidences)/len(confidences)*100:.1f}%")
    
    print(f"\n📍 Predictions by Location (top 10):")
    for loc, count in sorted(predictions_by_location.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   {loc}: {count}")
    
    print_header("GRAFANA QUERIES TO USE")
    print("Copy these queries into your Grafana dashboard:\n")
    print("1. Total Predictions per Location:")
    print("   sum(rain_predictions_total) by (location)\n")
    print("2. Prediction Confidence (95th percentile):")
    print("   histogram_quantile(0.95, sum(rate(rain_prediction_confidence_bucket[5m])) by (le))\n")
    print("3. Prediction Success Rate:")
    print("   sum(rate(http_requests_total{status=\"200\"}[5m])) / sum(rate(http_requests_total[5m]))\n")
    print("4. Input Humidity (Data Drift):")
    print("   avg(rain_input_humidity) by (location)\n")
    print("5. Error Rate (5xx):")
    print("   sum(rate(http_requests_total{status=~\"5..\"}[5m])) / sum(rate(http_requests_total[5m]))\n")
    
    print(f"🏙️  Locations used in this test run:")
    print(f"   {', '.join(sorted(locations))}")
    
    return results

def run_burst_test(num_requests=50):
    """Runs a burst of random requests."""
    # Select random 10 cities for this test run
    locations = get_random_locations()
    
    print_header("BURST TEST")
    print(f"🚀 Sending {num_requests} random requests in parallel...")
    print(f"🎲 Using {len(locations)} random locations: {', '.join(locations[:5])}...")
    
    payloads = [generate_random_payload(locations) for _ in range(num_requests)]
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_requests) as executor:
        results = list(executor.map(send_prediction, payloads))
    
    elapsed = time.time() - start_time
    
    successful = sum(1 for r in results if r["success"])
    failed = num_requests - successful
    
    print(f"\n📊 Results:")
    print(f"   Duration: {elapsed:.2f} seconds")
    print(f"   Throughput: {num_requests/elapsed:.1f} req/s")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate random test data for Grafana dashboards")
    parser.add_argument("-d", "--duration", type=int, default=5, help="Duration in minutes (default: 5)")
    parser.add_argument("-r", "--rate", type=int, default=2, help="Requests per second (default: 2)")
    parser.add_argument("-b", "--burst", type=int, default=0, help="Run burst test with N requests instead")
    
    args = parser.parse_args()
    
    print("="*70)
    print("🧪 GRAFANA DATA GENERATOR")
    print("="*70)
    print()
    print("This tool generates random prediction data for your Grafana dashboards.")
    print("It simulates real-world traffic with random locations and weather conditions.")
    print(f"📊 Total cities available: {len(ALL_LOCATIONS)}")
    print(f"🎲 Each test run uses 10 random cities to distribute load")
    print()
    
    if args.burst > 0:
        run_burst_test(args.burst)
    else:
        run_data_generator(
            duration_minutes=args.duration,
            requests_per_second=args.rate
        )
    
    print("\n✅ Done! Check your Grafana dashboard for updated metrics.")
    print("   Dashboard URL: http://34.247.195.225:3000")
    print()
    print("💡 Tip: Run again to get different random cities:")
    print("   python3 tests/grafana_test.py -d 5 -r 2")
