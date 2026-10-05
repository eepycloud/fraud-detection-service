"""FastAPI service that scores a single card transaction for fraud."""
from datetime import date
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, NaiveDatetime

from features import CATEGORIES, build_features

ROOT = Path(__file__).resolve().parents[1]
bundle = joblib.load(ROOT / "models" / "baseline.joblib")
MODEL = bundle["model"]
THRESHOLD = bundle["threshold"]

Category = Literal[tuple(CATEGORIES)]


class Transaction(BaseModel):
    trans_date_trans_time: NaiveDatetime
    amt: float = Field(gt=0, description="Transaction amount")
    category: Category
    gender: Literal["M", "F"]
    dob: date = Field(description="Cardholder date of birth")
    lat: float = Field(ge=-90, le=90, description="Cardholder latitude")
    long: float = Field(ge=-180, le=180, description="Cardholder longitude")
    merch_lat: float = Field(ge=-90, le=90)
    merch_long: float = Field(ge=-180, le=180)
    city_pop: int = Field(ge=0)

    model_config = {
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


app = FastAPI(
    title="Fraud Detection Service",
    description="Scores credit card transactions with a LightGBM model.",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score", response_model=Score)
def score(tx: Transaction):
    X = build_features(pd.DataFrame([tx.model_dump()]))
    prob = float(MODEL.predict_proba(X)[:, 1][0])
    return Score(
        fraud_probability=round(prob, 6),
        is_fraud=prob >= THRESHOLD,
        threshold=round(THRESHOLD, 4),
    )
