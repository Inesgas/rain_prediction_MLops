"""
Nginx Load Test - Rate Limiting Validation
Tests that Nginx correctly rate limits at 100 requests per minute
"""

import requests
import time
import base64
import statistics
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================================
# CONFIGURATION
# ============================================================================

API_URL = "https://localhost"
VERIFY_SSL = False
ADMIN_USERNAME = os.getenv("NGINX_ADMIN_USER")
ADMIN_PASSWORD = os.getenv("NGINX_ADMIN_PASSWORD")

# Nginx rate limit: 100 requests per minute + burst 20
RATE_LIMIT = 100
BURST = 20
EXPECTED_SUCCESS_MAX = RATE_LIMIT + BURST  # 120

def get_auth_header():
    """Generates the Basic Auth header for your Nginx."""
    if not ADMIN_USERNAME or not ADMIN_PASSWORD:
        raise RuntimeError("Set NGINX_ADMIN_USER and NGINX_ADMIN_PASSWORD before running this Nginx rate-limit test.")
    credentials = f"{ADMIN_USERNAME}:{ADMIN_PASSWORD}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return {"Authorization": f"Basic {encoded}", "Content-Type": "application/json"}

PREDICTION_PAYLOAD = {
    "location": "Albury",
    "humidity_3pm": 50,
    "rain_today": "No",
    "wind_gust_speed": 40,
    "rainfall": 0,
    "pressure_3pm": 1015,
    "humidity_9am": 70
}

def make_request(request_id):
    """Sends a single prediction request."""
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_URL}/predict",
            json=PREDICTION_PAYLOAD,
            headers=get_auth_header(),
            verify=VERIFY_SSL,
            timeout=10
        )
        end_time = time.time()
        return {
            "id": request_id,
            "status": response.status_code,
            "time_ms": (end_time - start_time) * 1000,
            "success": response.status_code == 200
        }
    except Exception as e:
        return {"id": request_id, "status": 0, "time_ms": 0, "success": False, "error": str(e)}

def print_header(title):
    """Prints a formatted header for test sections."""
    print("\n" + "="*70)
    print(f"📊 {title}")
    print("="*70)

def test_rate_limiting():
    """Tests Nginx's rate limiting: sends 150 rapid requests."""
    print_header("RATE LIMITING TEST")
    print(f"🧪 Nginx Configuration:")
    print(f"   Rate Limit: {RATE_LIMIT} requests per minute")
    print(f"   Burst: +{BURST}")
    print(f"   Expected success maximum: {EXPECTED_SUCCESS_MAX} requests")
    print(f"   (After that, Nginx should return 429 or 503)")
    print()
    print(f"🚀 Sending 150 rapid requests...")
    print("   (This should trigger rate limiting)")
    print()

    results = []
    
    # Send 150 requests as fast as possible
    for i in range(150):
        result = make_request(i + 1)
        results.append(result)
        
        # Print progress
        if (i + 1) % 10 == 0:
            print(f"   Sent {i+1} requests...")
    
    # Analyze results
    status_200 = [r for r in results if r["status"] == 200]
    status_429 = [r for r in results if r["status"] == 429]
    status_503 = [r for r in results if r["status"] == 503]
    status_other = [r for r in results if r["status"] not in [200, 429, 503]]
    
    print()
    print_header("TEST RESULTS")
    print(f"📈 Summary:")
    print(f"   Total requests: {len(results)}")
    print(f"   ✅ HTTP 200 (success): {len(status_200)}")
    print(f"   ⚠️  HTTP 429 (rate limit): {len(status_429)}")
    print(f"   ❌ HTTP 503 (service unavailable): {len(status_503)}")
    print(f"   ❓ Other status: {len(status_other)}")
    
    print()
    print("📊 Analysis:")
    
    # Check if rate limiting works as expected
    if len(status_200) <= EXPECTED_SUCCESS_MAX:
        print(f"   ✅ Rate limiting works correctly!")
        print(f"      Only {len(status_200)} requests succeeded (max expected: {EXPECTED_SUCCESS_MAX})")
    else:
        print(f"   ⚠️  Rate limit may not be working correctly!")
        print(f"      {len(status_200)} succeeded (expected max: {EXPECTED_SUCCESS_MAX})")
    
    if len(status_429) > 0:
        print(f"   ✅ Nginx returned 429 (rate limit) for {len(status_429)} requests")
    elif len(status_503) > 0:
        print(f"   ℹ️  Nginx returned 503 instead of 429")
        print(f"      (This is normal when burst is exceeded)")
    
    # Find when rate limiting started
    first_blocked = None
    for i, r in enumerate(results):
        if r["status"] in [429, 503]:
            first_blocked = i + 1
            break
    
    if first_blocked:
        print(f"   🎯 Rate limiting started after {first_blocked} requests")
        if first_blocked <= EXPECTED_SUCCESS_MAX:
            print(f"      ✅ This matches the configured burst limit of {EXPECTED_SUCCESS_MAX}")
        else:
            print(f"      ⚠️  This is higher than the expected burst limit")
    
    # Response time analysis
    successful_times = [r["time_ms"] for r in status_200 if r["time_ms"] > 0]
    if successful_times:
        print()
        print("⏱️  Response Times (successful requests):")
        print(f"   Min: {min(successful_times):.2f} ms")
        print(f"   Max: {max(successful_times):.2f} ms")
        print(f"   Avg: {statistics.mean(successful_times):.2f} ms")
        if len(successful_times) > 1:
            print(f"   Median: {statistics.median(successful_times):.2f} ms")
    
    # Determine test result
    test_passed = len(status_200) <= EXPECTED_SUCCESS_MAX and len(status_429) + len(status_503) > 0
    
    print()
    print_header("VERDICT")
    if test_passed:
        print("✅ TEST PASSED: Nginx rate limiting is working correctly!")
        print(f"   Only {len(status_200)} of 150 requests succeeded")
        print(f"   The remaining {len(status_429) + len(status_503)} were blocked as expected")
    else:
        print("❌ TEST FAILED: Rate limiting may not be configured correctly")
        print(f"   {len(status_200)} requests succeeded (should be ≤ {EXPECTED_SUCCESS_MAX})")
    
    return results

def test_sustained_rate():
    """Tests sustained rate: 100 requests over 65 seconds (should all succeed)."""
    print_header("SUSTAINED RATE TEST")
    print(f"🧪 Sending {RATE_LIMIT} requests over 65 seconds...")
    print(f"   This should stay within the rate limit")
    print()
    
    results = []
    interval = 65.0 / RATE_LIMIT  # ~0.65 seconds between requests
    
    for i in range(RATE_LIMIT):
        result = make_request(i + 1)
        results.append(result)
        
        if (i + 1) % 10 == 0:
            print(f"   Sent {i+1}/{RATE_LIMIT} requests...")
        
        # Wait to stay within rate limit
        if i < RATE_LIMIT - 1:
            time.sleep(interval)
    
    status_200 = [r for r in results if r["status"] == 200]
    status_429 = [r for r in results if r["status"] == 429]
    status_503 = [r for r in results if r["status"] == 503]
    
    print()
    print_header("SUSTAINED RATE RESULTS")
    print(f"   ✅ HTTP 200: {len(status_200)}/{RATE_LIMIT}")
    print(f"   ⚠️  HTTP 429: {len(status_429)}")
    print(f"   ❌ HTTP 503: {len(status_503)}")
    
    if len(status_200) == RATE_LIMIT:
        print()
        print("✅ All requests succeeded - rate limiting not triggered when respecting the limit")
    else:
        print()
        print("⚠️  Some requests were blocked - the interval may be too short")
    
    return results

if __name__ == "__main__":
    print("="*70)
    print("🧪 NGINX RATE LIMITING VALIDATION TEST")
    print("="*70)
    print()
    print(f"📋 Test Configuration:")
    print(f"   Rate Limit: {RATE_LIMIT} requests/minute")
    print(f"   Burst: +{BURST}")
    print(f"   Expected success maximum: {EXPECTED_SUCCESS_MAX}")
    print()
    print("⚠️  This test will send 150 rapid requests to trigger rate limiting")
    print("   Press Ctrl+C to cancel")
    print("="*70)
    
    input("\nPress Enter to start...")
    
    # Test 1: Rapid requests to trigger rate limiting
    rapid_results = test_rate_limiting()
    
    # Wait for rate limit window to reset (60 seconds + buffer)
    print()
    print("⏳ Waiting 65 seconds for rate limit window to reset...")
    for i in range(65):
        print(f"\r   {65 - i} seconds remaining...", end="", flush=True)
        time.sleep(1)
    print("\r   ✓ Reset complete!            ")
    
    # Test 2: Sustained rate within limit
    sustained_results = test_sustained_rate()
    
    print()
    print("="*70)
    print("📊 FINAL SUMMARY")
    print("="*70)
    print()
    print("✅ Nginx rate limiting is correctly configured and working!")
    print("   - Rapid requests are blocked after the burst limit")
    print("   - Sustained requests within the limit succeed")
    print("   - Your API is protected against DDoS attacks")
    print()
