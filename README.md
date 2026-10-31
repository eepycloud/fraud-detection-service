# Fraud Detection Service

Real-time credit card fraud detection. A LightGBM model trained on about 1.85M transactions, served through a FastAPI scoring endpoint.

**Live demo:** https://fraud-detection-service-srsh.onrender.com/docs
**Health:** https://fraud-detection-service-srsh.onrender.com/health

Hosted on Render Free. The service wakes on request after inactivity.

This project grew out of a fraud monitoring dashboard I built in Alteryx and Tableau during my training at Jordan Kuwait Bank ([dashboard](https://public.tableau.com/views/credit-card-fraud-bi/Dashboard), [repo](https://github.com/eepycloud/credit-card-fraud-bi)). The dashboard showed *where* fraud happens. This service takes the same data and scores individual transactions as they arrive.

> **Data note:** all work uses the public, simulated Sparkov dataset from Kaggle. No real bank data is used.

## Status

| Step | Work | Status |
| --- | --- | --- |
| 1 | Baseline model with precision, recall, and PR-AUC on the fraud class | Done |
| 2 | FastAPI scoring endpoint | Done |
| 2 | Docker image | Done (built and smoke-tested) |
| 3 | Free Render deployment with GitHub Actions CI checks | Live (31 Oct 2026) |
| 4 | Replay transactions as a live stream and store scores | Planned |
| 5 | Drift and performance monitoring dashboard | Planned |

## Results

Scored once on the held-out test file (555,719 transactions, 0.39% fraud):

| Metric | Value |
| --- | --- |
| PR-AUC | 0.871 |
| ROC-AUC | 0.997 |
| Precision | 87.5% |
| Recall | 76.4% |
| Decision threshold | 0.3482 |

Out of 2,145 fraudulent transactions, the model caught 1,638 and missed 507, with 234 false alarms.

**Why PR-AUC and not accuracy:** with fewer than 1 in 200 transactions being fraud, a model that flags nothing is over 99% accurate, and ROC-AUC looks near perfect for almost any model. PR-AUC shows how well the model finds the rare fraud cases.

### Gender as a feature

| Version | PR-AUC | Precision | Recall |
| --- | --- | --- | --- |
| With gender (used) | 0.871 | 87.5% | 76.4% |
| Without gender | 0.844 | 81.9% | 75.5% |

Gender is kept here because it improves results on this dataset. A production bank model would need a fairness review before using a protected attribute like this.

## How it works

**Data:** Kaggle Sparkov credit card transactions. `fraudTrain.csv` (1,296,675 rows) and `fraudTest.csv` (555,719 rows).

**Features (9):** transaction amount, hour, day of week, month, cardholder age, cardholder-to-merchant distance (haversine, km), city population, merchant category, and gender.

**Training:**
- LightGBM (learning rate 0.05, 63 leaves), early stopping at 222 trees.
- Time-based validation: the last 20% of the training period is held out for early stopping and for choosing the threshold (best F1). The model is never evaluated on transactions older than the ones it learned from.
- The test file is scored only once, at the end.

**Serving:** the API calls the same `build_features` function used in training, so features are computed identically in both places and training/serving mismatch can't creep in. Inputs are validated with Pydantic.

### Feature importance (gain)

| Feature | Share |
| --- | --- |
| Amount | 41.9% |
| Hour | 27.4% |
| Category | 9.5% |
| Age | 6.6% |
| City population | 6.2% |
| Gender | 3.7% |
| Month | 2.2% |
| Distance | 1.7% |
| Day of week | 1.0% |

Amount and hour dominate, which reflects how the simulator generates fraud. Cardholder-to-merchant distance was a headline view in the original monitoring dashboard but turned out to have almost no predictive value. A pattern that looks striking on a dashboard doesn't always help a model.

## Project structure

| Path | Purpose |
| --- | --- |
| `src/features.py` | Feature engineering shared by training and the API |
| `src/train.py` | Trains the model, picks the threshold, writes metrics and the model file |
| `src/app.py` | FastAPI app with `GET /health` and `POST /score` |
| `models/metrics.json` | Test metrics from the final run |
| `requirements.txt` | Python dependencies |

## Running it locally

1. Download the [Credit Card Transactions Fraud Detection dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) from Kaggle and put `fraudTrain.csv` and `fraudTest.csv` in a `data/` folder.

2. Set up the environment (Windows Command Prompt):

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

3. Train the model (writes `models/baseline.joblib`):

```
python src\train.py
```

4. Start the API:

```
uvicorn app:app --app-dir src --reload
```

5. Open http://127.0.0.1:8000/docs to try `POST /score` in the browser.

## Running with Docker

The service is trained on a public simulated dataset. No bank data is included.

Install and start Docker Desktop with Linux containers enabled. The repository includes the small saved model trained on a public simulated dataset, so a fresh clone can build without downloading the training CSV files.

From the project root (Windows Command Prompt):

```cmd
docker build -t fraud-detection-service:local .
docker run --detach --name fraud-api --publish 127.0.0.1:8000:8000 fraud-detection-service:local
curl.exe --fail http://127.0.0.1:8000/health
```

Open http://127.0.0.1:8000/docs and try the example transaction. The health
endpoint should return `{"status":"ok"}`. The existing example should score
approximately 0.000022, below the 0.3482 fraud threshold.

```cmd
docker inspect --format="{{.State.Health.Status}}" fraud-api
docker logs fraud-api
docker stop fraud-api
docker rm fraud-api
```

The image runs as an unprivileged user. Its build context includes only API
source, dependencies, and the saved model; datasets, local environments, and
Git history are excluded. The port is bound to this computer only.

## Free deployment on Render

The deployment target is Render's Free web service plan rather than Azure.
No student account is needed. Free services sleep after 15 minutes without
requests, so the next request can take about a minute to wake the API.
This is a public portfolio demo on a public simulated dataset, not a bank service.

1. Create a Render account and connect this GitHub repository.
2. Create a Blueprint from `render.yaml`, or create a Web Service using Docker.
3. Select **Free** (512 MB), port `8000`, and health check `/health`.
   Do not select a paid instance or add a database.
4. Enable auto-deploy **After CI Checks Pass** for the deployment branch.
5. Once live, open the service URL followed by `/docs` and run:

```cmd
python tests\smoke_test.py https://fraud-detection-service-srsh.onrender.com
```

GitHub Actions builds the Docker image, runs it with a 512 MB memory limit,
and tests health, scoring, and invalid inputs. Render rebuilds and deploys
commits after those checks pass. `render.yaml` records the service settings.
The saved model is included in Git; raw transaction CSV files remain excluded.

Free-tier limits: https://render.com/docs/free

## Verification

Verified on 31 Oct 2026: the Linux container passed all smoke checks under a 512 MB memory cap, reported healthy, and used approximately 119 MiB at the post-test measurement.


Run the smoke test against a local API or container:

```cmd
python tests\smoke_test.py http://127.0.0.1:8000
```

It verifies the documented baseline probability, an increased-risk transaction,
health, and rejection of invalid amounts, categories, coordinates, gender,
timezone-aware timestamps, and missing fields.

## API

`POST /score` takes a single transaction (time, amount, category, gender, date of birth, cardholder and merchant coordinates, city population) and returns:

```json
{
  "fraud_probability": 0.000022,
  "is_fraud": false,
  "threshold": 0.3482
}
```

See `/docs` for the full input schema.

## License

MIT
