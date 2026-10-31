"""FastAPI service that scores a single card transaction for fraud."""
from contextlib import asynccontextmanager
from datetime import date
import json
import os
import time
import uuid
import threading
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, NaiveDatetime, model_validator
from store import Store
from monitoring import summarize

from features import CATEGORIES, build_features

ROOT = Path(__file__).resolve().parents[1]
bundle = joblib.load(ROOT / "models" / "baseline.joblib")
MODEL = bundle["model"]
THRESHOLD = bundle["threshold"]

Category = Literal[tuple(CATEGORIES)]


class Transaction(BaseModel):
    trans_date_trans_time: NaiveDatetime
    amt: float = Field(gt=0, le=1e9, description="Transaction amount")
    category: Category
    gender: Literal["M", "F"]
    dob: date = Field(description="Cardholder date of birth")
    lat: float = Field(ge=-90, le=90, description="Cardholder latitude")
    long: float = Field(ge=-180, le=180, description="Cardholder longitude")
    merch_lat: float = Field(ge=-90, le=90)
    merch_long: float = Field(ge=-180, le=180)
    city_pop: int = Field(ge=0, le=1000000000)

    @model_validator(mode="after")
    def valid_birth_date(self):
        if not 1900 <= self.trans_date_trans_time.year <= 2100:
            raise ValueError("Transaction year must be between 1900 and 2100")
        if (self.trans_date_trans_time.date() - self.dob).days > 120 * 365.25:
            raise ValueError("Cardholder age cannot exceed 120 years")
        if self.dob > self.trans_date_trans_time.date():
            raise ValueError("Date of birth must precede the transaction")
        return self

    model_config = {
        "allow_inf_nan": False,
        "json_schema_extra": {
            "examples": [
                {
                    "trans_date_trans_time": "2019-01-01T00:00:18",
                    "amt": 4.97,
                    "category": "misc_net",
                    "gender": "F",
                    "dob": "1988-03-09",
                    "lat": 36.0788,
                    "long": -81.1781,
                    "merch_lat": 36.011293,
                    "merch_long": -82.048315,
                    "city_pop": 3495,
                }
            ]
        }
    }


class Score(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold: float



REPLAY_LOCK = threading.Lock()
STORE = Store(os.environ.get("SCORES_DB", str(ROOT / "runtime" / "scores.db")))
REFERENCE = json.loads((ROOT / "models" / "reference.json").read_text(encoding="utf-8"))
DEMO = json.loads((ROOT / "demo" / "transactions.json").read_text(encoding="utf-8"))

def score_rows(transactions, source, ids, labels=None):
    X = build_features(pd.DataFrame(transactions))
    started = time.perf_counter()
    probabilities = MODEL.predict_proba(X)[:, 1]
    elapsed = (time.perf_counter()-started)*1000
    rows=[]
    for i, tx in enumerate(transactions):
        features=X.iloc[i].to_dict()
        features={k:(v.item() if hasattr(v,"item") else v) for k,v in features.items()}
        rows.append({"id":ids[i],"event_time":str(tx["trans_date_trans_time"]),"amount":float(tx["amt"]),
                     "hour":int(features["hour"]),"category":tx["category"],"probability":float(probabilities[i]),
                     "prediction":int(probabilities[i]>=THRESHOLD),"label":None if labels is None else int(labels[i]),
                     "source":source,"latency_ms":elapsed if len(transactions)==1 else None,
                     "features":json.dumps(features)})
    STORE.add(rows)
    return probabilities

@asynccontextmanager
async def lifespan(app):
    if not STORE.recent("demo", 1):
        score_rows(DEMO,"demo",[f"demo:{i:05d}" for i in range(len(DEMO))],[r["is_fraud"] for r in DEMO])
    yield

app = FastAPI(title="Fraud Detection Service",
              description="Fraud scoring, transaction replay, and monitoring on a public simulated dataset.",
              version="1.0.0", lifespan=lifespan)

@app.get("/", include_in_schema=False)
@app.get("/dashboard", include_in_schema=False)
def index():
    return FileResponse(ROOT / "src" / "dashboard.html")

@app.get("/health")
def health():
    with STORE.connection() as db:
        db.execute("SELECT 1")
    return {"status":"ok"}

@app.post("/score", response_model=Score)
def score(tx: Transaction):
    probability=float(score_rows([tx.model_dump()],"live",[uuid.uuid4().hex])[0])
    return Score(fraud_probability=round(probability,6),is_fraud=probability>=THRESHOLD,threshold=round(THRESHOLD,4))

@app.get("/api/monitor")
def monitor(source: Literal["demo","live","replay"]="demo"):
    result=summarize(STORE.recent(source),REFERENCE,float(THRESHOLD))
    result.update({"source":source,"model_version":"lightgbm-222-v1","storage_ephemeral":bool(os.environ.get("RENDER")),
                   "max_records_per_source":5000,"demo_size":len(DEMO)})
    return result


@app.post("/api/replay")
def replay(batch: int=Query(default=50,ge=1,le=50)):
    with REPLAY_LOCK:
        start,end=STORE.reserve_demo(len(DEMO),batch)
        selected=DEMO[start:end]
        try:
            if selected:
                score_rows(selected,"replay",[f"browser-replay:{i:05d}" for i in range(start,end)],[r["is_fraud"] for r in selected])
        except Exception:
            with STORE.connection() as db:
                db.execute("INSERT OR REPLACE INTO state VALUES ('demo_cursor',?)",(start,))
            raise
    return {"processed":end-start,"cursor":end,"total":len(DEMO),"complete":end==len(DEMO)}
