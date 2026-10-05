"""Feature engineering shared by model training and the scoring API."""
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "is_fraud"

# Fixed list so the category encoding is identical in training and serving.
CATEGORIES = [
    "entertainment", "food_dining", "gas_transport", "grocery_net",
    "grocery_pos", "health_fitness", "home", "kids_pets", "misc_net",
    "misc_pos", "personal_care", "shopping_net", "shopping_pos", "travel",
]

FEATURES = [
    "amt", "hour", "day_of_week", "month", "age",
    "distance_km", "city_pop", "category","gender",
]


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometres between two points."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw transaction rows into the model's input columns."""
    ts = pd.to_datetime(df["trans_date_trans_time"])
    dob = pd.to_datetime(df["dob"])

    out = pd.DataFrame(index=df.index)
    out["amt"] = df["amt"].astype(float)
    out["hour"] = ts.dt.hour
    out["day_of_week"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["age"] = (ts - dob).dt.days / 365.25
    out["distance_km"] = haversine_km(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )
    out["city_pop"] = df["city_pop"]
    out["category"] = pd.Categorical(df["category"], categories=CATEGORIES)
    out["gender"] = (df["gender"] == "M").astype(int)
    return out[FEATURES]


if __name__ == "__main__":
    # Sanity check: confirm the dataset and the features look right.
    data_dir = Path(__file__).resolve().parents[1] / "data"
    for name in ("fraudTrain.csv", "fraudTest.csv"):
        df = pd.read_csv(data_dir / name)
        X = build_features(df)
        print(f"\n{name}")
        print(f"  rows: {len(df):,}   fraud rate: {df[TARGET].mean():.3%}")
        print(f"  raw columns: {list(df.columns)}")
        print(f"  missing values in features: {int(X.isna().sum().sum())}")
        print(X.head(3).to_string())
