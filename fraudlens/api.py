"""FastAPI service.

Run:   uvicorn fraudlens.api:app --reload
Docs:  http://127.0.0.1:8000/docs
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, create_model

from .config import RAW_COLUMNS
from .service import FraudService

# Request schema with all 30 raw inputs; built in a loop instead of 28 lines of V-fields.
_fields = {c: (float, Field(...)) for c in RAW_COLUMNS}
_fields["Time"] = (float, Field(..., ge=0, description="Seconds since first transaction"))
_fields["Amount"] = (float, Field(..., ge=0, description="Transaction amount"))
Transaction = create_model("Transaction", **_fields)


class Reason(BaseModel):
    feature: str
    value: float
    impact: float
    text: str


class ScoreResponse(BaseModel):
    fraud_probability: float
    decision: str
    threshold: float
    top_reasons: list[Reason]
    latency_ms: float


state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["service"] = FraudService()      # load the model once, not per request
    except FileNotFoundError:
        state["service"] = None                # API still starts; /health reports it
    yield
    state.clear()


app = FastAPI(title="FraudLens", version="1.0",
              description="Explainable credit-card fraud scoring", lifespan=lifespan)


def _service() -> FraudService:
    svc = state.get("service")
    if svc is None:
        raise HTTPException(503, "Model not trained yet. Run: python -m fraudlens.train")
    return svc


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state.get("service") is not None}


@app.get("/model-info")
def model_info():
    svc = _service()
    return {"threshold": svc.threshold, "metrics": svc.metrics}


@app.post("/score", response_model=ScoreResponse)
def score(tx: Transaction):
    return _service().score([tx.model_dump()])[0]


@app.post("/score/batch", response_model=list[ScoreResponse])
def score_batch(txs: list[Transaction]):
    if not 1 <= len(txs) <= 1000:
        raise HTTPException(422, "Send between 1 and 1000 transactions")
    return _service().score([t.model_dump() for t in txs])
