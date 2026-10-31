"""Offline drift-triggered candidate retraining. Never replaces the serving model."""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import average_precision_score
from features import build_features
from monitoring import summarize
from store import Store

ROOT=Path(__file__).resolve().parents[1]
def once(args):
    reference=json.loads((ROOT/"models/reference.json").read_text())
    bundle=joblib.load(ROOT/"models/baseline.joblib")
    report=summarize(Store(args.db).recent(args.source),reference,bundle["threshold"])
    shifted=any(d["psi"] is not None and d["psi"]>=.2 for d in report["drift"])
    if not (args.force or shifted):
        print("No candidate trained: no detected shift with at least 200 observations.")
        return
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out=ROOT/"models/candidates"/stamp
    out.mkdir(parents=True,exist_ok=False)
    subprocess.run([sys.executable,str(ROOT/"src/train.py"),"--output-dir",str(out),"--threads","4"],cwd=ROOT,check=True)
    # Compare on the original chronological validation period; never use test
    # performance as the promotion gate.
    frame=pd.read_csv(ROOT/"data/fraudTrain.csv").sort_values("trans_date_trans_time")
    validation=frame.iloc[int(len(frame)*.8):]
    X=build_features(validation); y=validation["is_fraud"]
    candidate=joblib.load(out/"baseline.joblib")
    old=float(average_precision_score(y,bundle["model"].predict_proba(X)[:,1]))
    new=float(average_precision_score(y,candidate["model"].predict_proba(X)[:,1]))
    decision={"created_at":stamp,"trigger":"forced" if args.force else "distribution_shift",
              "drift":report["drift"],"baseline_validation_pr_auc":old,"candidate_validation_pr_auc":new,
              "eligible_for_manual_review":new>=old,"promoted":False,
              "note":"Candidate only. Review temporal evaluation, fairness, and provenance before replacing the serving model."}
    (out/"review.json").write_text(json.dumps(decision,indent=2))
    print(json.dumps({"candidate_directory":str(out),"baseline_validation_pr_auc":old,
                      "candidate_validation_pr_auc":new,"promoted":False},indent=2))
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db",default=str(ROOT/"runtime/scores.db"))
    p.add_argument("--source",choices=["demo","live","replay"],default="replay")
    p.add_argument("--force",action="store_true",help="Train a candidate even without drift")
    p.add_argument("--watch-seconds",type=int,default=0,help="Optional foreground monitoring interval; no startup task is installed")
    args=p.parse_args()
    if args.watch_seconds and args.watch_seconds<60:p.error("Watch interval must be at least 60 seconds")
    # Watch mode retrains only when NEW observations have arrived.
    fingerprint=None
    while True:
        rows=Store(args.db).recent(args.source)
        current=tuple(r["id"] for r in rows)
        if current!=fingerprint:
            once(args); fingerprint=current
        if not args.watch_seconds: break
        time.sleep(args.watch_seconds)
if __name__=="__main__": main()
