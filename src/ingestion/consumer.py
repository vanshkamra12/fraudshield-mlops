"""
Fraud scoring consumer — reads transactions from Kafka, calls FastAPI, publishes results.

Usage:
    python -m src.ingestion.consumer                    # run forever
    python -m src.ingestion.consumer --max 1000         # stop after 1000 predictions
"""
import argparse
import json
import logging
import time

import requests
from confluent_kafka import Consumer, Producer, KafkaError, KafkaException

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [CONSUMER] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

KAFKA_BROKER   = "localhost:9092"
INPUT_TOPIC    = "transactions"
OUTPUT_TOPIC   = "predictions"
SCORING_API    = "http://localhost:8000"
GROUP_ID       = "fraudshield-scorer"


def make_consumer(broker: str, group: str) -> Consumer:
    return Consumer({
        "bootstrap.servers": broker,
        "group.id": group,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
        "auto.commit.interval.ms": 5000,
    })


def make_producer(broker: str) -> Producer:
    return Producer({
        "bootstrap.servers": broker,
        "acks": 1,
    })


def score_transaction(api_url: str, payload: dict) -> dict | None:
    """Call /predict on the FastAPI service. Returns None on error."""
    # strip internal field before sending to API
    actual_label = payload.pop("_actual_label", None)

    try:
        resp = requests.post(f"{api_url}/predict", json=payload, timeout=5)
        resp.raise_for_status()
        result = resp.json()

        # record ground truth feedback immediately (simulates near-real-time labelling)
        if actual_label is not None:
            try:
                requests.post(
                    f"{api_url}/feedback",
                    params={
                        "transaction_id": payload["transaction_id"],
                        "actual_label": actual_label,
                    },
                    timeout=2,
                )
            except Exception:
                pass  # feedback is best-effort

        return result

    except requests.exceptions.Timeout:
        logger.warning(f"Timeout scoring {payload.get('transaction_id')}")
    except requests.exceptions.RequestException as e:
        logger.error(f"API error: {e}")
    return None


def run(
    broker: str = KAFKA_BROKER,
    api_url: str = SCORING_API,
    max_messages: int | None = None,
) -> None:
    consumer = make_consumer(broker, GROUP_ID)
    out_producer = make_producer(broker)
    consumer.subscribe([INPUT_TOPIC])

    logger.info(f"Subscribed to [{INPUT_TOPIC}] | Scoring via {api_url}")
    logger.info(f"Publishing results to [{OUTPUT_TOPIC}]")

    processed = 0
    fraud_detected = 0
    errors = 0
    start = time.time()

    try:
        while True:
            msg = consumer.poll(timeout=1.0)

            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    logger.info("Reached end of partition")
                else:
                    raise KafkaException(msg.error())
                continue

            try:
                payload = json.loads(msg.value().decode())
            except json.JSONDecodeError as e:
                logger.error(f"Bad JSON: {e}")
                errors += 1
                continue

            result = score_transaction(api_url, payload)

            if result is None:
                errors += 1
                continue

            # publish result to output topic
            out_producer.produce(
                OUTPUT_TOPIC,
                key=result["transaction_id"],
                value=json.dumps(result).encode(),
            )
            out_producer.poll(0)

            processed += 1
            if result["predicted_label"] == 1:
                fraud_detected += 1

            if processed % 200 == 0:
                elapsed = time.time() - start
                rate = processed / elapsed
                logger.info(
                    f"Processed {processed:,}  "
                    f"fraud_detected={fraud_detected} ({fraud_detected/processed:.1%})  "
                    f"errors={errors}  rate={rate:.1f} txn/s"
                )

            if max_messages and processed >= max_messages:
                logger.info(f"Reached max_messages={max_messages}. Stopping.")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        consumer.close()
        out_producer.flush(timeout=10)
        elapsed = time.time() - start
        logger.info(
            f"Stopped. Processed {processed:,} in {elapsed:.1f}s | "
            f"fraud_detected={fraud_detected} | errors={errors}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FraudShield scoring consumer")
    parser.add_argument("--broker", default=KAFKA_BROKER)
    parser.add_argument("--api", default=SCORING_API)
    parser.add_argument("--max", type=int, default=None)
    args = parser.parse_args()

    run(broker=args.broker, api_url=args.api, max_messages=args.max)
