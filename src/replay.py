"""Replay a CSV in timestamp order, score through HTTP, and journal locally."""
import argparse
import csv
import json
import time
import uuid
from urllib.request import Request, urlopen
from store import Store
from features import build_features
import pandas as pd

FIELDS=("trans_date_trans_time","amt","category","gender","dob","lat","long","merch_lat","merch_long","city_pop")
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv",required=True)
    parser.add_argument("--api",default="http://127.0.0.1:8000")
    parser.add_argument("--db",default="runtime/scores.db")
    parser.add_argument("--limit",type=int,default=1000)
    parser.add_argument("--rate",type=float,default=10,help="Transactions per second; 0 means no delay")
    args=parser.parse_args()
    if args.limit<1 or args.rate<0: parser.error("limit must be positive and rate nonnegative")
    rows=[]
    with open(args.csv,newline="",encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append(row)
            if len(rows)>=args.limit: break
    rows.sort(key=lambda r:r["trans_date_trans_time"])
    store=Store(args.db)
    run=uuid.uuid4().hex
    for index,row in enumerate(rows):
        tick=time.perf_counter()
        tx={k:row[k] for k in FIELDS}
        for key in ("amt","lat","long","merch_lat","merch_long"): tx[key]=float(tx[key])
        tx["city_pop"]=int(tx["city_pop"])
        req=Request(args.api.rstrip("/")+"/score",data=json.dumps(tx).encode(),headers={"Content-Type":"application/json"})
        with urlopen(req,timeout=90) as response: result=json.load(response)
        elapsed=(time.perf_counter()-tick)*1000
        features=build_features(pd.DataFrame([tx])).iloc[0].to_dict()
        features={k:(v.item() if hasattr(v,"item") else v) for k,v in features.items()}
        store.add([{"id":f"{run}:{index:08d}","event_time":tx["trans_date_trans_time"],"amount":tx["amt"],
                    "hour":features["hour"],"category":tx["category"],"probability":result["fraud_probability"],
                    "prediction":int(result["is_fraud"]),"label":int(row["is_fraud"]) if row.get("is_fraud") in ("0","1") else None,
                    "source":"replay","latency_ms":elapsed,"features":json.dumps(features)}])
        if (index+1)%100==0: print(f"Stored {index+1}/{len(rows)}")
        if args.rate: time.sleep(max(0,1/args.rate-(time.perf_counter()-tick)))
    print(f"Completed {len(rows)} transactions. SQLite journal: {store.path}")
if __name__=="__main__": main()
