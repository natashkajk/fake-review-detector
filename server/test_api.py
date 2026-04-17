"""
Fake Review Detector - API Test Script
Test the /analyze endpoint with sample reviews
"""

import requests
import json
import time

# API Configuration
API_BASE_URL = "http://localhost:8000"
ANALYZE_ENDPOINT = f"{API_BASE_URL}/analyze"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"

# Test reviews
TEST_REVIEWS = {
    "likely_fake": [
        "This is the BEST product ever!!! Must buy now!!! Amazing quality!!!",
        "I received this product for free in exchange for my honest review. It's absolutely fantastic and I highly recommend it to everyone!",
        "CLICK HERE to buy!!! Limited time offer!!! Don't miss out!!! Best deal ever!!!",
        "Great product good quality works well as described very satisfied happy with love this item",
        "OMG!!!! This is AMAZING!!!! Best purchase EVER!!!! 10/10!!!!"
    ],
    "likely_genuine": [
        "The product arrived on time and works as expected. Nothing special, but does the job.",
        "I've been using this for about a month now. The build quality is decent, though the buttons feel a bit cheap. Overall satisfied with the purchase.",
        "Good value for the price. Had some issues with setup but customer service helped resolve them within a day.",
        "Not the best product I've used, but certainly not the worst. Pros: affordable, easy to use. Cons: battery life could be better.",
        "Bought this as a gift for my sister. She seems to like it. Shipping was fast and packaging was secure."
    ]
}


def test_health():
    """Test the health endpoint"""
    print("=" * 60)
    print("Testing Health Endpoint")
    print("=" * 60)
    
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_analyze(text: str, expected_type: str = None):
    """Test the analyze endpoint with a single review"""
    print(f"\nTesting: {text[:60]}...")
    
    try:
        start_time = time.time()
        response = requests.post(
            ANALYZE_ENDPOINT,
            json={"text": text},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        elapsed = time.time() - start_time
        
        print(f"  Status: {response.status_code}")
        print(f"  Time: {elapsed:.3f}s")
        
        if response.status_code == 200:
            result = response.json()
            print(f"  Prediction: {result['prediction'].upper()}")
            print(f"  Confidence: {result['confidence']:.2%}")
            print(f"  Suspicious phrases: {len(result['suspicious_phrases'])}")
            
            if result['suspicious_phrases']:
                for phrase in result['suspicious_phrases'][:3]:
                    print(f"    - {phrase[:50]}")
            
            if expected_type and result['prediction'] != expected_type:
                print(f"  ⚠️  Expected {expected_type}, got {result['prediction']}")
            
            return result
        else:
            print(f"  Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"  Error: {e}")
        return None


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("FAKE REVIEW DETECTOR - API TEST SUITE")
    print("=" * 60)
    
    # Test health
    if not test_health():
        print("\n❌ Health check failed. Is the server running?")
        print(f"   Make sure the server is running at {API_BASE_URL}")
        return
    
    print("\n" + "=" * 60)
    print("Testing LIKELY FAKE Reviews")
    print("=" * 60)
    
    fake_results = []
    for review in TEST_REVIEWS["likely_fake"]:
        result = test_analyze(review, expected_type="fake")
        if result:
            fake_results.append(result)
        time.sleep(0.5)  # Small delay between requests
    
    print("\n" + "=" * 60)
    print("Testing LIKELY GENUINE Reviews")
    print("=" * 60)
    
    genuine_results = []
    for review in TEST_REVIEWS["likely_genuine"]:
        result = test_analyze(review, expected_type="genuine")
        if result:
            genuine_results.append(result)
        time.sleep(0.5)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    fake_correct = sum(1 for r in fake_results if r['prediction'] == 'fake')
    genuine_correct = sum(1 for r in genuine_results if r['prediction'] == 'genuine')
    
    print(f"Fake reviews detected: {fake_correct}/{len(fake_results)}")
    print(f"Genuine reviews detected: {genuine_correct}/{len(genuine_results)}")
    
    if fake_results:
        avg_fake_conf = sum(r['confidence'] for r in fake_results) / len(fake_results)
        print(f"Avg confidence (fake): {avg_fake_conf:.2%}")
    
    if genuine_results:
        avg_genuine_conf = sum(r['confidence'] for r in genuine_results) / len(genuine_results)
        print(f"Avg confidence (genuine): {avg_genuine_conf:.2%}")
    
    total_tests = len(fake_results) + len(genuine_results)
    total_correct = fake_correct + genuine_correct
    
    if total_tests > 0:
        accuracy = total_correct / total_tests
        print(f"\nOverall accuracy: {accuracy:.1%} ({total_correct}/{total_tests})")
    
    print("\n" + "=" * 60)
    print("Tests completed!")
    print("=" * 60)


def interactive_test():
    """Interactive testing mode"""
    print("\n" + "=" * 60)
    print("INTERACTIVE TEST MODE")
    print("=" * 60)
    print("Enter a review to analyze (or 'quit' to exit):\n")
    
    while True:
        text = input("> ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            break
        
        if len(text) < 5:
            print("Text too short. Please enter at least 5 characters.")
            continue
        
        test_analyze(text)
        print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_test()
    else:
        run_all_tests()