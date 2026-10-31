"""Population Stability Index and metrics for labeled replay data."""
import numpy as np
from sklearn.metrics import average_precision_score

def distribution(values, spec):
    if spec["kind"] == "numeric":
        counts = np.histogram(values, bins=spec["bins"])[0].astype(float)
    else:
        counts = np.array([sum(str(v)==c for v in values) for c in spec["categories"]],dtype=float)
    return counts

def psi(expected, observed):
    e = np.asarray(expected,dtype=float)+0.5
    o = np.asarray(observed,dtype=float)+0.5
    e /= e.sum()
    o /= o.sum()
    return float(np.sum((o-e)*np.log(o/e)))

def summarize(rows, reference, threshold):
    n = len(rows)
    labeled = [r for r in rows if r["label"] is not None]
    tp = sum(r["label"]==1 and r["prediction"]==1 for r in labeled)
    fp = sum(r["label"]==0 and r["prediction"]==1 for r in labeled)
    fn = sum(r["label"]==1 and r["prediction"]==0 for r in labeled)
    tn = len(labeled)-tp-fp-fn
    labels = [r["label"] for r in labeled]
    ap = float(average_precision_score(labels,[r["probability"] for r in labeled])) if len(set(labels))==2 else None
    latency = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
    drift=[]
    for name,spec in reference.items():
        values=[r["probability"] if name=="fraud_probability" else r["features"][name] for r in rows]
        observed=distribution(values,spec)
        value=psi(spec["counts"],observed) if n>=200 else None
        drift.append({"feature":name,"psi":value,"status":"insufficient data" if value is None else ("shift detected" if value>=0.2 else "stable"),
                      "reference":spec["counts"],"observed":observed.astype(int).tolist(),"bins":spec.get("labels",spec.get("categories"))})
    by_day={}; by_category={}
    for r in rows:
        day=r["event_time"][:10]
        d=by_day.setdefault(day,{"day":day,"count":0,"flagged":0})
        d["count"]+=1; d["flagged"]+=r["prediction"]
        c=by_category.setdefault(r["category"],{"category":r["category"],"count":0,"flagged":0})
        c["count"]+=1; c["flagged"]+=r["prediction"]
    return {"count":n,"flagged":sum(r["prediction"] for r in rows),
            "flag_rate":sum(r["prediction"] for r in rows)/n if n else 0,
            "labeled":len(labeled),"actual_fraud":sum(labels),
            "precision":tp/(tp+fp) if tp+fp else None,"recall":tp/(tp+fn) if tp+fn else None,
            "pr_auc":ap,"confusion":{"tp":tp,"fp":fp,"fn":fn,"tn":tn},
            "latency_p95_ms":float(np.percentile(latency,95)) if latency else None,
            "threshold":threshold,"drift":drift,"timeline":sorted(by_day.values(),key=lambda x:x["day"]),
            "categories":sorted(by_category.values(),key=lambda x:x["count"],reverse=True),
            "recent":[{k:r[k] for k in ("event_time","amount","category","probability","prediction","label","source")} for r in rows[-20:]][::-1]}
