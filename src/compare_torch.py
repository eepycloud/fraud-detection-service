"""Optional CPU MLP benchmark using the same features and temporal split."""
import argparse
import copy
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.metrics import average_precision_score,precision_recall_curve,precision_score,recall_score
from features import FEATURES,build_features,CATEGORIES

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs",type=int,default=15)
    args=p.parse_args()
    torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(4)
    frame=pd.read_csv(ROOT/"data/fraudTrain.csv").sort_values("trans_date_trans_time")
    test=pd.read_csv(ROOT/"data/fraudTest.csv").sort_values("trans_date_trans_time")
    X=build_features(frame);cut=int(len(frame)*.8)
    numeric=[c for c in FEATURES if c!="category"]
    pre=ColumnTransformer([("numeric",StandardScaler(),numeric),
                          ("category",OneHotEncoder(categories=[CATEGORIES],handle_unknown="ignore",sparse_output=False),["category"])])
    train=torch.tensor(pre.fit_transform(X.iloc[:cut]).astype("float32"))
    val=torch.tensor(pre.transform(X.iloc[cut:]).astype("float32"))
    y=torch.tensor(frame["is_fraud"].iloc[:cut].to_numpy(dtype="float32")).reshape(-1,1)
    yval=frame["is_fraud"].iloc[cut:].to_numpy()
    network=nn.Sequential(nn.Linear(train.shape[1],64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    optimizer=torch.optim.Adam(network.parameters(),lr=.001)
    loss=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(y)-y.sum().item())/y.sum().item()]))
    best=-1;weights=None;patience=0
    for epoch in range(args.epochs):
        network.train();order=torch.randperm(len(train))
        for indices in order.split(4096):
            optimizer.zero_grad()
            error=loss(network(train[indices]),y[indices])
            error.backward();optimizer.step()
        network.eval()
        with torch.no_grad():prob=torch.sigmoid(network(val)).flatten().numpy()
        ap=float(average_precision_score(yval,prob))
        print(f"Epoch {epoch+1}: validation PR-AUC {ap:.4f}",flush=True)
        if ap>best:
            best=ap;weights=copy.deepcopy(network.state_dict());patience=0
        else:
            patience+=1
            if patience>=3:break
    network.load_state_dict(weights);network.eval()
    with torch.no_grad():
        prob=torch.sigmoid(network(val)).flatten().numpy()
    precision,recall,thresholds=precision_recall_curve(yval,prob)
    f1=2*precision*recall/(precision+recall+1e-12)
    threshold=float(thresholds[np.argmax(f1[:-1])])
    with torch.no_grad():
        scores=torch.sigmoid(network(torch.tensor(pre.transform(build_features(test)).astype("float32")))).flatten().numpy()
    prediction=scores>=threshold
    metrics={"model":"PyTorch MLP (64,32)","features":FEATURES,"validation_pr_auc":best,
             "threshold":threshold,"pr_auc":float(average_precision_score(test["is_fraud"],scores)),
             "precision":float(precision_score(test["is_fraud"],prediction,zero_division=0)),
             "recall":float(recall_score(test["is_fraud"],prediction,zero_division=0)),
             "test_rows":len(test),"epochs_run":epoch+1,"torch_version":torch.__version__,
             "note":"Comparison only; serving model remains LightGBM. Threshold and early stopping use the chronological validation period."}
    (ROOT/"models/torch_metrics.json").write_text(json.dumps(metrics,indent=2))
    torch.save({"state_dict":weights,"input_dim":train.shape[1],"threshold":threshold},ROOT/"models/comparison.pt")
    joblib.dump(pre,ROOT/"models/torch_preprocessor.joblib")
    print(json.dumps(metrics,indent=2))
if __name__=="__main__":main()
