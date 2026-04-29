from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class TransactionInput(BaseModel):
    transaction_id: str = Field(..., description="Unique transaction ID")

    amount: float = Field(..., description="Transaction amount")
    card1: float = Field(..., description="Card number feature")
    card2: float = Field(..., description="Card type")
    card3: float = Field(..., description="Card issuer")
    card4: str = Field(..., description="Card brand (Visa, Mastercard, etc.)")
    card5: float = Field(..., description="Card category")
    card6: str = Field(..., description="Card issuer code")

    addr1: float = Field(..., description="Billing address 1")
    addr2: float = Field(..., description="Billing address 2")
    dist1: float = Field(..., description="Distance to first address")

    ProductCD: str = Field(..., description="Product code")
    P_emaildomain: str = Field(..., description="Purchaser email domain")
    R_emaildomain: str = Field(..., description="Recipient email domain")

    DeviceType: str = Field(..., description="Device type (mobile/desktop)")
    TransactionDT: int = Field(..., description="Time since epoch")

    C1: Optional[float] = None
    C2: Optional[float] = None
    C3: Optional[float] = None
    C4: Optional[float] = None
    C5: Optional[float] = None
    C6: Optional[float] = None
    C7: Optional[float] = None
    C8: Optional[float] = None
    C9: Optional[float] = None
    C10: Optional[float] = None
    C11: Optional[float] = None
    C12: Optional[float] = None
    C13: Optional[float] = None
    C14: Optional[float] = None

    D1: Optional[float] = None
    D2: Optional[float] = None
    D3: Optional[float] = None
    D4: Optional[float] = None
    D5: Optional[float] = None
    D10: Optional[float] = None
    D11: Optional[float] = None
    D15: Optional[float] = None

    M1: Optional[str] = None
    M2: Optional[str] = None
    M3: Optional[str] = None
    M4: Optional[str] = None
    M5: Optional[str] = None
    M6: Optional[str] = None
    M7: Optional[str] = None
    M8: Optional[str] = None
    M9: Optional[str] = None

    id_01: Optional[float] = None
    id_02: Optional[float] = None
    id_05: Optional[float] = None
    id_06: Optional[float] = None
    id_11: Optional[float] = None
    id_13: Optional[float] = None
    id_17: Optional[float] = None
    id_19: Optional[float] = None
    id_20: Optional[float] = None

    V12: Optional[float] = None
    V13: Optional[float] = None
    V17: Optional[float] = None
    V18: Optional[float] = None
    V33: Optional[float] = None
    V34: Optional[float] = None
    V35: Optional[float] = None
    V36: Optional[float] = None
    V37: Optional[float] = None
    V38: Optional[float] = None
    V39: Optional[float] = None
    V40: Optional[float] = None
    V42: Optional[float] = None
    V43: Optional[float] = None
    V44: Optional[float] = None
    V45: Optional[float] = None
    V51: Optional[float] = None
    V52: Optional[float] = None
    V53: Optional[float] = None
    V54: Optional[float] = None
    V55: Optional[float] = None
    V56: Optional[float] = None
    V57: Optional[float] = None
    V70: Optional[float] = None
    V74: Optional[float] = None
    V75: Optional[float] = None
    V76: Optional[float] = None
    V77: Optional[float] = None
    V78: Optional[float] = None
    V79: Optional[float] = None
    V80: Optional[float] = None
    V81: Optional[float] = None
    V82: Optional[float] = None
    V83: Optional[float] = None
    V86: Optional[float] = None
    V87: Optional[float] = None
    V91: Optional[float] = None
    V92: Optional[float] = None
    V93: Optional[float] = None
    V94: Optional[float] = None
    V95: Optional[float] = None
    V96: Optional[float] = None
    V97: Optional[float] = None
    V126: Optional[float] = None
    V127: Optional[float] = None
    V128: Optional[float] = None
    V130: Optional[float] = None
    V131: Optional[float] = None
    V307: Optional[float] = None
    V308: Optional[float] = None

class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_score: float
    predicted_label: int
    threshold: float = 0.7379

    class Config:
        from_attributes = True
