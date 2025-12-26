"""
Test ECU Agent Docker API

Simple script to test the deployed Docker container.

Usage:
    python test_docker_api.py
"""

import requests
import json


API_URL = "http://localhost:8080/invocations"


def test_health():
    """Test health endpoint."""
    print("="*80)
    print("🏥 Testing Health Check")
    print("="*80)
    
    try:
        response = requests.get("http://localhost:8080/health", timeout=5)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Health check passed!")
        else:
            print("❌ Health check failed!")
        print()
    except Exception as e:
        print(f"❌ Error: {e}\n")


def test_single_prediction():
    """Test single prediction."""
    print("="*80)
    print("🔮 Testing Single Prediction")
    print("="*80)
    
    payload = {
        "inputs": {
            "query": "What is the operating temperature range for ECU-700?"
        }
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}\n")
    
    try:
        response = requests.post(
            API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            print("\n✅ Prediction successful!")
        else:
            print(f"❌ Error: {response.text}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()


def test_batch_prediction():
    """Test batch prediction."""
    print("="*80)
    print("📦 Testing Batch Prediction")
    print("="*80)
    
    payload = {
        "inputs": [
            {"query": "What is ECU-700?"},
            {"query": "What communication protocols does ECU-800 support?"},
            {"query": "Compare ECU-700 and ECU-800 specifications"}
        ]
    }
    
    print(f"Request: {len(payload['inputs'])} queries\n")
    
    try:
        response = requests.post(
            API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            results = response.json()
            print(f"Received {len(results.get('predictions', []))} predictions\n")
            
            for i, pred in enumerate(results.get('predictions', []), 1):
                print(f"{i}. Query: {pred.get('query', 'N/A')[:50]}...")
                print(f"   Answer: {pred.get('answer', 'N/A')[:100]}...\n")
            
            print("✅ Batch prediction successful!")
        else:
            print(f"❌ Error: {response.text}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()


def main():
    """Run all tests."""
    print("\n🧪 ECU Agent Docker API Tests\n")
    
    # Run tests
    test_health()
    test_single_prediction()
    test_batch_prediction()
    
    print("="*80)
    print("✅ All tests completed!")
    print("="*80)


if __name__ == "__main__":
    main()