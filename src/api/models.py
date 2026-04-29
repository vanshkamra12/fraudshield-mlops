from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String, nullable=False, unique=True)

    fraud_score = Column(Float, nullable=False)
    predicted_label = Column(Integer, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    actual_label = Column(Integer, nullable=True)

    __table_args__ = (
        {"schema": "public"},
    )
