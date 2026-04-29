"""
Example FraudShield API client
Demonstrates how to score transactions and record feedback
"""

import requests
import json
import time
from datetime import datetime

API_URL = "http://localhost:8000"

def check_health():
    """Verify API and model are loaded"""
    response = requests.get(f"{API_URL}/health")
    return response.json()

def predict_fraud(transaction_id: str, features: dict) -> dict:
    """
    Score a transaction for fraud

    Args:
        transaction_id: Unique transaction identifier
        features: Dict of raw transaction features (amount, card1, etc.)

    Returns:
        Dict with fraud_score, predicted_label, threshold
    """
    payload = {"transaction_id": transaction_id, **features}

    response = requests.post(f"{API_URL}/predict", json=payload)
    response.raise_for_status()
    return response.json()

def record_feedback(transaction_id: str, actual_label: int):
    """
    Record ground truth label after transaction outcome is known

    Args:
        transaction_id: Must match a previous prediction
        actual_label: 0=legitimate, 1=fraud
    """
    response = requests.post(
        f"{API_URL}/feedback",
        params={"transaction_id": transaction_id, "actual_label": actual_label}
    )
    response.raise_for_status()
    return response.json()

if __name__ == "__main__":
    # Check health
    print("Checking API health...")
    health = check_health()
    print(f"  Status: {health['status']}")
    print(f"  Model loaded: {health['model_loaded']}")
    print(f"  Pipeline loaded: {health['pipeline_loaded']}")

    if not health['model_loaded'] or not health['pipeline_loaded']:
        print("ERROR: Model or pipeline not loaded")
        exit(1)

    # Example transaction features (minimal required fields)
    features = {
        "amount": 150.50,
        "card1": 1234.0,
        "card2": 5678.0,
        "card3": 9.0,
        "card4": "visa",
        "card5": 0.0,
        "card6": "american_express",
        "addr1": 100.0,
        "addr2": 200.0,
        "dist1": 50.0,
        "ProductCD": "C",
        "P_emaildomain": "gmail.com",
        "R_emaildomain": "yahoo.com",
        "DeviceType": "desktop",
        "TransactionDT": int(time.time()),
    }

    # Score transaction
    print("\nScoring transaction...")
    txn_id = f"example_txn_{int(time.time())}"
    result = predict_fraud(txn_id, features)

    print(f"  Transaction ID: {result['transaction_id']}")
    print(f"  Fraud Score: {result['fraud_score']:.4f}")
    print(f"  Predicted Label: {result['predicted_label']} ({'FRAUD' if result['predicted_label'] == 1 else 'LEGITIMATE'})")
    print(f"  Threshold: {result['threshold']:.4f}")

    # Simulate feedback after some delay
    print("\nWaiting 5 seconds to simulate transaction outcome...")
    time.sleep(5)

    print("Recording feedback...")
    feedback = record_feedback(txn_id, actual_label=0)
    print(f"  {feedback['status']}")

    print("\nExample complete!")
