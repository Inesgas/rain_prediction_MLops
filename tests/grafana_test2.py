"""
Grafana Dashboard Data Generator - Throttled Version
Sends prediction requests at a paced rate to test/populate Grafana
dashboards. Adjust DURATION_MINUTES and REQUESTS_PER_SECOND below to
control load. Nginx's configured limit is rate=100r/m (~1.67 req/s)
with burst=20 — rates above that will gradually exhaust the burst
buffer and trigger HTTP 503 responses from the apilimit zone.
"""

import requests
import time
import random
import base64
import os
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ============================================================================
# CONFIGURATION - adjust these two values to control the test
# ============================================================================

DURATION_MINUTES = 0.5      # how long the generator runs
REQUESTS_PER_SECOND = 3.0   # how many requests are sent per second

# Nginx's real limit, for reference/warnings only (does not affect sending)
NGINX_SAFE_REQUESTS_PER_SECOND = 100 / 60  # ~1.67 req/s

API_URL = "https://localhost"
VERIFY_SSL = False
ADMIN_USERNAME = os.getenv("NGINX_ADMIN_USER", "admin")
ADMIN_PASSWORD = os.getenv("NGINX_ADMIN_PASSWORD", "admin")

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
    "Walpole", "Watsonia", "Woomera",
]

WEATHER_SCENARIOS = [
    {"name": "sunny", "humidity": (20, 40), "rain_today": "No", "pressure": (1015, 1025)},
    {"name": "cloudy", "humidity": (50, 70), "rain_today": "No", "pressure": (1010, 1020)},
    {"name": "rainy", "humidity": (75, 95), "rain_today": "Yes", "pressure": (998, 1010)},
    {"name": "storm", "humidity": (80, 98), "rain_today": "Yes", "pressure": (985, 1000)},
    {"name": "humid", "humidity": (70, 90), "rain_today": "No", "pressure": (1005, 1015)},
    {"name": "dry", "humidity": (10, 30), "rain_today": "No", "pressure": (1020, 1030)},
]


def get_random_locations():
    return random.sample(ALL_LOCATIONS, 10)


def get_auth_header():
    credentials = f"{ADMIN_USERNAME}:{ADMIN_PASSWORD}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return {"Authorization": f"Basic {encoded}", "Content-Type": "application/json"}


def generate_random_payload(locations):
    location = random.choice(locations)
    scenario = random.choice(WEATHER_SCENARIOS)
    humidity = random.uniform(*scenario["humidity"])
    pressure = random.uniform(*scenario["pressure"])
    rainfall = random.uniform(0, 15) if scenario["rain_today"] == "Yes" else random.uniform(0, 2)
    humidity_9am = max(0, min(100, humidity + random.uniform(-15, 5)))
    return {
        "location": location,
        "humidity_3pm": round(humidity, 1),
        "rain_today": scenario["rain_today"],
        "wind_gust_speed": round(random.uniform(10, 80), 1),
        "rainfall": round(rainfall, 1),
        "pressure_3pm": round(pressure, 1),
        "humidity_9am": round(humidity_9am, 1),
    }


def send_prediction(payload):
    try:
        start = time.time()
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            headers=get_auth_header(),
            verify=VERIFY_SSL,
            timeout=10,
        )
        elapsed_ms = (time.time() - start) * 1000
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "status": response.status_code,
                "time_ms": elapsed_ms,
                "location": payload["location"],
                "prediction": data.get("prediction"),
                "confidence": data.get("confidence"),
            }
        return {"success": False, "status": response.status_code, "time_ms": elapsed_ms}
    except Exception as e:
        return {"success": False, "status": 0, "time_ms": 0, "error": str(e)}


def print_header(title):
    print("\n" + "=" * 70)
    print(f"📊 {title}")
    print("=" * 70)


def run_throttled_generator():
    if REQUESTS_PER_SECOND > NGINX_SAFE_REQUESTS_PER_SECOND:
        print(f"⚠️  Rate {REQUESTS_PER_SECOND} req/s exceeds Nginx's real limit "
              f"(~{NGINX_SAFE_REQUESTS_PER_SECOND:.2f} req/s). The burst buffer (20) "
              f"will gradually be exhausted and some requests will get HTTP 503.")

    locations = get_random_locations()
    total_requests = round(DURATION_MINUTES * 60 * REQUESTS_PER_SECOND)
    interval = 1.0 / REQUESTS_PER_SECOND

    print_header("THROTTLED GRAFANA DATA GENERATOR")
    print(f"📍 API URL: {API_URL}")
    print(f"⏱️  Duration: {DURATION_MINUTES} minutes")
    print(f"🚀 Rate: {REQUESTS_PER_SECOND} req/s")
    print(f"📨 Total requests planned: {total_requests}")
    print(f"🎲 Locations for this run: {', '.join(locations[:5])}... ({len(locations)-5} more)")
    print("=" * 70)

    results = []
    successful = 0
    failed = 0
    start_time = time.time()

    print(f"\n🔄 Sending {REQUESTS_PER_SECOND} request(s) every 1 s "
          f"({total_requests} requests total over {DURATION_MINUTES * 60:.0f}s) ...")
    print("   Press Ctrl+C to stop early\n")

    try:
        for i in range(total_requests):
            payload = generate_random_payload(locations)
            result = send_prediction(payload)
            results.append(result)

            if result["success"]:
                successful += 1
            else:
                failed += 1

            if (i + 1) % 20 == 0 or (i + 1) == total_requests:
                print(f"   ✓ {i+1}/{total_requests} sent — {successful} ok, {failed} failed")

            if i < total_requests - 1:
                time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped by user")

    elapsed = time.time() - start_time

    print_header("SUMMARY")
    print(f"⏱️  Duration: {elapsed:.1f}s")
    print(f"📨 Total requests: {len(results)}")
    print(f"✅ Successful: {successful} ({successful/len(results)*100:.1f}%)")
    print(f"❌ Failed: {failed} ({failed/len(results)*100:.1f}%)")
    if failed == 0:
        print("\n✅ No requests were rate-limited — traffic stayed within Nginx's configured limit.")
    else:
        print("\n⚠️  Some requests were rejected — Nginx's burst buffer was exhausted at this rate.")

    return results


if __name__ == "__main__":
    run_throttled_generator()
    print("\n✅ Done! Check your Grafana dashboard for updated metrics.")
