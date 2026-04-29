import os
import joblib
import numpy as np
import pandas as pd
import logging
from datetime import datetime

from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
from starlette.routing import Mount

from src.api.models import Base, Prediction
from src.api.schemas import TransactionInput, PredictionResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.7379"))
FEATURE_PIPELINE_PATH = os.getenv("FEATURE_PIPELINE_PATH", "models/feature_pipeline.joblib")
MODEL_PATH = os.getenv("MODEL_PATH", "models/lgbm_model.txt")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://fraudshield:fraudshield123@localhost:5432/fraudshield")

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

# ── Prometheus metrics ────────────────────
PREDICTIONS_TOTAL   = Counter("fraudshield_predictions_total", "Total predictions", ["label"])
FRAUD_SCORE_HIST    = Histogram("fraudshield_fraud_score", "Fraud score distribution",
                                buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
PREDICT_LATENCY     = Histogram("fraudshield_predict_latency_seconds", "Prediction latency")
FRAUD_RATE_GAUGE    = Gauge("fraudshield_fraud_rate", "Rolling fraud rate (last 1000 predictions)")

app = FastAPI(
    title="FraudShield Scoring",
    version="1.0",
    description="Real-time fraud detection scoring service",
    routes=[Mount("/metrics", make_asgi_app())],
)

try:
    feature_pipeline = joblib.load(FEATURE_PIPELINE_PATH)
    import lightgbm as lgb
    model = lgb.Booster(model_file=MODEL_PATH)
    logger.info(f"Loaded model from {MODEL_PATH}")
    logger.info(f"Loaded pipeline from {FEATURE_PIPELINE_PATH}")
except Exception as e:
    logger.error(f"Failed to load model/pipeline: {e}")
    model = None
    feature_pipeline = None


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "pipeline_loaded": feature_pipeline is not None,
        "threshold": THRESHOLD,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(txn: TransactionInput):
    if model is None or feature_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    db = SessionLocal()
    import time
    start = time.perf_counter()
    try:
        txn_dict = txn.dict()
        transaction_id = txn_dict.pop("transaction_id")

        df = pd.DataFrame([txn_dict])
        df_transformed = feature_pipeline.transform(df)

        fraud_score = float(model.predict(df_transformed.values.astype(np.float32))[0])
        predicted_label = 1 if fraud_score >= THRESHOLD else 0

        prediction = Prediction(
            transaction_id=transaction_id,
            fraud_score=fraud_score,
            predicted_label=predicted_label,
            created_at=datetime.utcnow(),
        )
        db.add(prediction)
        db.commit()
        db.refresh(prediction)

        # ── update Prometheus ─────────────
        PREDICTIONS_TOTAL.labels(label=str(predicted_label)).inc()
        FRAUD_SCORE_HIST.observe(fraud_score)
        PREDICT_LATENCY.observe(time.perf_counter() - start)

        # rolling fraud rate from last 1000 predictions
        try:
            recent = db.query(Prediction).order_by(Prediction.id.desc()).limit(1000).all()
            if recent:
                fraud_rate = sum(p.predicted_label for p in recent) / len(recent)
                FRAUD_RATE_GAUGE.set(fraud_rate)
        except Exception:
            pass

        return PredictionResponse(
            transaction_id=transaction_id,
            fraud_score=fraud_score,
            predicted_label=predicted_label,
            threshold=THRESHOLD,
        )

    except Exception as e:
        db.rollback()
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
    finally:
        db.close()


@app.post("/feedback")
async def feedback(transaction_id: str, actual_label: int):
    db = SessionLocal()
    try:
        prediction = db.query(Prediction).filter(
            Prediction.transaction_id == transaction_id
        ).first()

        if not prediction:
            raise HTTPException(status_code=404, detail="Prediction not found")

        prediction.actual_label = actual_label
        db.commit()

        return {"status": "feedback recorded"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        db.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
