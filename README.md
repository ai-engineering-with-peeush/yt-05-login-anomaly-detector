# Login Anomaly Detector

> Companion code for: **[Build a Login Anomaly Detector in Python — No Labels, No Rules (End-to-End)](https://youtu.be/h3JQGq-ZA_I)**
> Part of the *Anomaly & Behavioral Detection in Production* series on [AI Engineering with Peeush](https://www.youtube.com/@AIEngineeringWithPeeush).

Build an unsupervised login anomaly detector in Python — no labels, no rules, end to end.

## What this builds

```
Raw Auth Logs → Feature Extraction → Isolation Forest → Scoring Gate → FastAPI
```

Three behavioral features catch three different anomaly shapes:

| Feature | Catches |
|---|---|
| `geo_velocity_kmh` | Impossible travel (point anomaly) |
| `device_novelty` + `hour_deviation` | New device at odd hour (contextual anomaly) |
| `burst_rate` | Credential-stuffing burst (collective anomaly) |

## Quickstart

```bash
pip install -r requirements.txt
make data    # generate synthetic auth logs → data/logs.csv + data/labels.csv
make eval    # train detector, score all events, print metrics
make serve   # start FastAPI scoring service (requires make eval first)
```

## Repo structure

```
src/
  models.py          # LoginEvent + UserHistory
  generate_logs.py   # synthetic data generator (pre-built)
  features.py        # feature extraction (3 live-coded + burst_rate pre-built)
  detector.py        # IsolationForest + scoring gate (built live on camera)
evals/
  run_eval.py        # honest evaluation against known attacks
serve.py             # FastAPI scoring endpoint
starter/
  features.py        # live-coding starting point (docstring shells)
  detector.py        # live-coding starting point (imports only)
```

## The honest evaluation

The eval harness opens `data/labels.csv` only at scoring time — the detector never sees labels during training or scoring. Results show precision/recall per attack type + alerts per 10k events + baseline comparison against a simple z-score on geo_velocity alone.

**Known miss:** `new_device_odd_hour` is partially caught. Users with irregular schedules (on-call, shift workers) have a wide hour distribution — a 2 AM login barely deviates from their median. The fix is per-user behavioral baselines, which is Episode 2.

## Series: Anomaly & Behavioral Detection in Production

- **Episode 1 (this repo):** Login anomaly detector — raw logs to scoring API
- **Episode 2:** Behavioral baselining — per-user adaptive thresholds
- **Episode 3:** Drift-aware retraining — keeping the detector current
