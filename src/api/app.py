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

# ordered list of the 113 engineered features the model was trained on
FEATURE_COLUMNS = [
    'ProductCD','card1','card2','card3','card4','card5','card6',
    'addr1','addr2','dist1','P_emaildomain','R_emaildomain',
    'C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14',
    'D1','D2','D3','D4','D5','D8','D10','D11','D15',
    'M1','M2','M3','M4','M5','M6','M7','M8','M9',
    'V12','V13','V17','V18','V33','V34','V35','V36','V37','V38','V39','V40',
    'V42','V43','V44','V45','V51','V52','V53','V54','V55','V56','V57','V70',
    'V74','V75','V76','V77','V78','V79','V80','V81','V82','V83','V86','V87',
    'V91','V92','V93','V94','V95','V96','V97','V126','V127','V128','V130','V131',
    'V307','V308',
    'id_01','id_02','id_05','id_06','id_09','id_10','id_11','id_13','id_14',
    'id_17','id_19','id_20','id_32',
    'hour_of_day','day_of_week','amt_log','amt_zscore','has_identity','device_mobile',
]

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


@app.post("/score", response_model=PredictionResponse)
async def score(payload: dict):
    """
    Score pre-engineered features directly — used by the Kafka consumer.
    Accepts transaction_id + the 113 engineered feature columns, skips pipeline.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    db = SessionLocal()
    import time
    start = time.perf_counter()
    try:
        transaction_id = payload.get("transaction_id", f"unknown_{int(time.time()*1000)}")
        payload.pop("_actual_label", None)  # strip internal field if present

        feature_values = [float(payload.get(col, 0.0) or 0.0) for col in FEATURE_COLUMNS]
        X = np.array(feature_values, dtype=np.float32).reshape(1, -1)

        fraud_score = float(model.predict(X)[0])
        predicted_label = 1 if fraud_score >= THRESHOLD else 0

        prediction = Prediction(
            transaction_id=transaction_id,
            fraud_score=fraud_score,
            predicted_label=predicted_label,
            created_at=datetime.utcnow(),
        )
        db.add(prediction)
        db.commit()

        PREDICTIONS_TOTAL.labels(label=str(predicted_label)).inc()
        FRAUD_SCORE_HIST.observe(fraud_score)
        PREDICT_LATENCY.observe(time.perf_counter() - start)

        try:
            recent = db.query(Prediction).order_by(Prediction.id.desc()).limit(1000).all()
            if recent:
                FRAUD_RATE_GAUGE.set(sum(p.predicted_label for p in recent) / len(recent))
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
        logger.error(f"Score error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
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
