"""
Transaction producer — reads from test.parquet and streams to Kafka.

Usage:
    python -m src.ingestion.producer                    # stream full test set at 10 txn/s
    python -m src.ingestion.producer --rate 50          # 50 transactions/second
    python -m src.ingestion.producer --max 1000         # stop after 1000 transactions
    python -m src.ingestion.producer --once             # send all rows, then exit
"""
import argparse
import json
import time
import logging

import numpy as np
import pandas as pd
from confluent_kafka import Producer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PRODUCER] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

KAFKA_BROKER = "localhost:9092"
TOPIC = "transactions"


def make_producer(broker: str = KAFKA_BROKER) -> Producer:
    return Producer({
        "bootstrap.servers": broker,
        "acks": "all",
        "retries": 3,
        "retry.backoff.ms": 200,
    })


def delivery_callback(err, msg):
    if err:
        logger.error(f"Delivery failed: {err}")


def stream_transactions(
    data_path: str = "data/processed/test.parquet",
    rate: float = 10.0,
    max_rows: int | None = None,
    once: bool = False,
    broker: str = KAFKA_BROKER,
) -> None:
    df = pd.read_parquet(data_path)

    # Keep isFraud for the feedback simulation, but don't send it in the payload
    labels = df.pop("isFraud").values if "isFraud" in df.columns else None

    if max_rows:
        df = df.head(max_rows)
        if labels is not None:
            labels = labels[:max_rows]

    producer = make_producer(broker)
    delay = 1.0 / rate
    total = len(df)

    logger.info(f"Streaming {total:,} transactions to [{TOPIC}] at {rate:.0f} txn/s")
    logger.info(f"Broker: {broker}")

    sent = 0
    fraud_sent = 0
    start = time.time()

    try:
        for i, (_, row) in enumerate(df.iterrows()):
            txn_id = f"txn_{i:08d}_{int(time.time() * 1000) % 100000}"

            payload = {
                "transaction_id": txn_id,
                **{k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                   for k, v in row.items()},
            }

            # include ground truth so the consumer can POST /feedback after scoring
            if labels is not None:
                payload["_actual_label"] = int(labels[i])

            producer.produce(
                TOPIC,
                key=txn_id,
                value=json.dumps(payload).encode(),
                callback=delivery_callback,
            )
            producer.poll(0)

            sent += 1
            if labels is not None and labels[i] == 1:
                fraud_sent += 1

            if sent % 500 == 0:
                elapsed = time.time() - start
                actual_rate = sent / elapsed
                logger.info(
                    f"Sent {sent:,}/{total:,}  "
                    f"fraud={fraud_sent} ({fraud_sent/sent:.1%})  "
                    f"rate={actual_rate:.1f} txn/s"
                )

            if not once:
                time.sleep(delay)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        producer.flush(timeout=10)
        elapsed = time.time() - start
        logger.info(
            f"Done. Sent {sent:,} transactions in {elapsed:.1f}s "
            f"(fraud: {fraud_sent}, {fraud_sent/max(sent,1):.1%})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FraudShield transaction producer")
    parser.add_argument("--rate", type=float, default=10.0, help="Transactions per second")
    parser.add_argument("--max", type=int, default=None, help="Max rows to send")
    parser.add_argument("--once", action="store_true", help="Send all rows once and exit")
    parser.add_argument("--broker", default=KAFKA_BROKER)
    parser.add_argument("--data", default="data/processed/test.parquet")
    args = parser.parse_args()

    stream_transactions(
        data_path=args.data,
        rate=args.rate,
        max_rows=args.max,
        once=args.once,
        broker=args.broker,
    )
