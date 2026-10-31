"""Build deterministic demo/reference samples from PUBLIC SIMULATED CSVs."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from features import CATEGORIES,build_features
from replay import FIELDS
from monitoring import distribution

ROOT=Path(__file__).resolve().parents[1]
def sample(path,n=5000):
    total=sum(len(chunk) for chunk in pd.read_csv(path,usecols=["is_fraud"],chunksize=100000))
    chosen=set(np.random.default_rng(42).choice(total,min(n,total),replace=False).tolist())
    parts=[];offset=0
    for chunk in pd.read_csv(path,chunksize=100000):
        mask=[i-offset for i in chosen if offset<=i<offset+len(chunk)]
        if mask: parts.append(chunk.iloc[sorted(mask)])
        offset+=len(chunk)
    return pd.concat(parts).sort_values("trans_date_trans_time")
def main():
    train=sample(ROOT/"data/fraudTrain.csv")
    test=sample(ROOT/"data/fraudTest.csv")
    X=build_features(train)
    model=joblib.load(ROOT/"models/baseline.joblib")["model"]
    specs={
        "amt":{"kind":"numeric","bins":[-1,10,25,50,100,250,500,1000,1000000000],"labels":["<10","10-25","25-50","50-100","100-250","250-500","500-1k","1k+"]},
        "hour":{"kind":"numeric","bins":[0,4,8,12,16,20,24],"labels":["00-03","04-07","08-11","12-15","16-19","20-23"]},
        "category":{"kind":"categorical","categories":CATEGORIES},
        "fraud_probability":{"kind":"numeric","bins":[-1,0.001,0.01,0.05,0.2,0.3482,0.7,1.01],"labels":["<.001",".001-.01",".01-.05",".05-.2",".2-.348",".348-.7",".7+"]}
    }
    for name,spec in specs.items():
        values=model.predict_proba(X)[:,1] if name=="fraud_probability" else X[name].tolist()
        spec["counts"]=distribution(values,spec).astype(int).tolist()
    (ROOT/"models/reference.json").write_text(json.dumps(specs),encoding="utf-8")
    demo=test[[*FIELDS,"is_fraud"]].to_dict(orient="records")
    (ROOT/"demo").mkdir(exist_ok=True)
    (ROOT/"demo/transactions.json").write_text(json.dumps(demo,separators=(",",":")),encoding="utf-8")
    print(f"Prepared {len(demo)} public simulated demo rows; {sum(r['is_fraud'] for r in demo)} labeled frauds")
if __name__=="__main__": main()
