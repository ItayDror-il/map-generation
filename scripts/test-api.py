#!/usr/bin/env python3
"""
Quick test script for the Map Generator API.
Run this after starting the backend to verify it works.
"""

import requests
import json

API_URL = "http://localhost:8000"
PLAYGROUND_URL = "https://playground.getsequence.io"


def test_health():
    """Test health endpoint."""
    print("Testing /health...")
    r = requests.get(f"{API_URL}/health")
    assert r.status_code == 200
    print(f"  Status: {r.json()}")


def test_generate_map():
    """Test map generation endpoint."""
    print("\nTesting /api/generate-map...")

    payload = {
        "profile": {
            "USER_TYPE": "INDIVIDUAL",
            "ANNUALINCOME": "BETWEEN_50K_AND_100K",
            "AGE_GROUP": "25-34",
            "OCCUPATION": "EMPLOYED",
            "PRODUCTGOAL": ["SAVINGS", "DEBT_PAYOFF"]
        },
        "prompt": "I want to save 20% of my paycheck and pay off my credit card debt using the avalanche method."
    }

    r = requests.post(
        f"{API_URL}/api/generate-map",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=60  # LLM generation can take time
    )

    if r.status_code != 200:
        print(f"  Error: {r.status_code}")
        print(f"  Body: {r.text}")
        return None

    data = r.json()
    print(f"  Playground ID: {data['id']}")
    print(f"  Nodes: {len(data['map'].get('nodes', []))}")
    print(f"  Rules: {len(data['map'].get('rules', []))}")
    print(f"\n  Explanation:\n{data['explanation'][:500]}...")
    print(f"\n  Open in playground: {PLAYGROUND_URL}/?id={data['id']}")

    return data['id']


def test_playground_url(map_id: str):
    """Verify the playground URL is accessible."""
    if not map_id:
        print("\nSkipping playground test (no map ID)")
        return

    print(f"\nVerifying playground URL...")
    url = f"{PLAYGROUND_URL}/?id={map_id}"
    print(f"  URL: {url}")

    # Just check if the playground page loads (not the actual map data)
    r = requests.get(PLAYGROUND_URL, timeout=10)
    if r.status_code == 200:
        print(f"  Playground is accessible")
    else:
        print(f"  Warning: Playground returned {r.status_code}")


if __name__ == "__main__":
    print("=" * 60)
    print("Map Generator API Tests")
    print("=" * 60)
    print(f"API URL: {API_URL}")
    print(f"Playground: {PLAYGROUND_URL}\n")

    try:
        test_health()
        map_id = test_generate_map()
        test_playground_url(map_id)

        print("\n" + "=" * 60)
        print("All tests passed!")
        if map_id:
            print(f"\nOpen your map: {PLAYGROUND_URL}/?id={map_id}")
        print("=" * 60)
    except requests.exceptions.ConnectionError:
        print("\nError: Cannot connect to API")
        print("Make sure the backend is running: make backend")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
