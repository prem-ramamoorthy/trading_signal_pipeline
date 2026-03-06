################################################################################
#  EUR/USD Candle Colour Predictor  —  Sample cURL Requests
#  Base URL: http://localhost:8000
#  Start server: python run.py serve  (or uvicorn api:app --reload --port 8000)
################################################################################


# ══════════════════════════════════════════════════════════════════════════════
#  GET /
#  Root — welcome message + route map
# ══════════════════════════════════════════════════════════════════════════════

curl -s http://localhost:8000/ | python3 -m json.tool


# ── Expected response ─────────────────────────────────────────────────────────
# {
#   "name": "EUR/USD Candle Colour Predictor",
#   "version": "1.0.0",
#   "docs": "/docs",
#   "routes": {
#     "GET  /health": "Liveness + artifact status",
#     "POST /predict": "Single prediction from JSON candles",
#     ...
#   }
# }


# ══════════════════════════════════════════════════════════════════════════════
#  GET /health
#  Liveness probe — artifact status, pipeline steps, uptime
# ══════════════════════════════════════════════════════════════════════════════

curl -s http://localhost:8000/health | python3 -m json.tool


# ── Expected response ─────────────────────────────────────────────────────────
# {
#   "status": "ok",
#   "artifacts_loaded": true,
#   "artifacts": {
#     "master_pipe": true,
#     "candle_colour_model": true,
#     "metrics": true
#   },
#   "model_steps": ["cleaning", "engineer", "warmup", "selector", "scaler"],
#   "n_features": 21,
#   "uptime_seconds": 4.2
# }


# ══════════════════════════════════════════════════════════════════════════════
#  GET /metrics
#  Stored validation + test evaluation metrics
# ══════════════════════════════════════════════════════════════════════════════

curl -s http://localhost:8000/metrics | python3 -m json.tool


# ── Expected response ─────────────────────────────────────────────────────────
# {
#   "features": ["returns", "log_returns", "hl_range", ...],
#   "n_features": 21,
#   "validation": {
#     "accuracy": 0.5158,
#     "f1": 0.3858,
#     "roc_auc": 0.5117
#   },
#   "test": {
#     "accuracy": 0.5161,
#     "f1": 0.4855,
#     "roc_auc": 0.5219
#   }
# }


# ══════════════════════════════════════════════════════════════════════════════
#  GET /features
#  Feature registry + Random Forest importances (ranked highest first)
# ══════════════════════════════════════════════════════════════════════════════

curl -s http://localhost:8000/features | python3 -m json.tool


# ── Expected response ─────────────────────────────────────────────────────────
# {
#   "n_features": 21,
#   "features": ["returns", "log_returns", "hl_range", ...],
#   "importances": {
#     "return_lag_1": 0.0821,
#     "atr_ratio":    0.0763,
#     "ema_ratio":    0.0701,
#     "rsi_14":       0.0698,
#     ...
#   }
# }


# ══════════════════════════════════════════════════════════════════════════════
#  GET /predict/latest
#  Predict next candle from the on-disk raw CSV (no body required)
#  Use for live polling / cron jobs
# ══════════════════════════════════════════════════════════════════════════════

curl -s "http://localhost:8000/predict/latest" | python3 -m json.tool

# ── With custom tail (use last 200 rows as context) ───────────────────────────
curl -s "http://localhost:8000/predict/latest?tail=200" | python3 -m json.tool

# ── With tail=150 (minimum safe context) ─────────────────────────────────────
curl -s "http://localhost:8000/predict/latest?tail=150" | python3 -m json.tool


# ── Expected response ─────────────────────────────────────────────────────────
# {
#   "colour":      "RED",
#   "green_prob":  0.486710,
#   "red_prob":    0.513290,
#   "confidence":  0.513290,
#   "signal":      0,
#   "last_candle": "2026-02-26T19:15:00+05:30",
#   "latency_ms":  312.4
# }

# ── Error: tail too small (< 120) — 422 ──────────────────────────────────────
curl -s "http://localhost:8000/predict/latest?tail=50" | python3 -m json.tool


# ══════════════════════════════════════════════════════════════════════════════
#  POST /predict
#  Single prediction from a JSON array of candles
#  Requires: at least 120 candles, newest last
#  Columns:  time (ISO-8601), open, high, low, close, Volume
# ══════════════════════════════════════════════════════════════════════════════

# ── Option A: inline minimal payload (3 candles shown, send ≥120 in practice) ─

curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "candles": [
      {"time":"2026-02-25T12:15:00+05:30","open":1.18020,"high":1.18064,"low":1.18018,"close":1.18053,"Volume":922},
      {"time":"2026-02-25T12:30:00+05:30","open":1.18052,"high":1.18080,"low":1.18042,"close":1.18046,"Volume":1316},
      {"time":"2026-02-25T12:45:00+05:30","open":1.18046,"high":1.18078,"low":1.18032,"close":1.18071,"Volume":987}
    ]
  }' | python3 -m json.tool

# NOTE: the above will return 422 (too few candles).
#       Use the file-based command below for a real prediction.

# ── Option B: load from candles.json file (recommended for ≥120 candles) ──────
#
# First generate the file:
#   python3 -c "
#   import pandas as pd, json
#   df = pd.read_csv('data/raw/OANDA_EURUSD_15.csv')
#   rows = df.tail(125).to_dict(orient='records')
#   json.dump({'candles': rows}, open('candles.json','w'))
#   print('candles.json created with', len(rows), 'candles')
#   "
#
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @candles.json | python3 -m json.tool


# ── Expected response ─────────────────────────────────────────────────────────
# {
#   "colour":      "RED",
#   "green_prob":  0.486710,
#   "red_prob":    0.513290,
#   "confidence":  0.513290,
#   "signal":      0,
#   "last_candle": "2026-02-26T19:15:00+05:30",
#   "latency_ms":  289.1
# }

# ── Error: too few candles (< 120) — 422 ─────────────────────────────────────
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"candles": [
    {"time":"2026-02-26T19:15:00+05:30","open":1.18032,"high":1.18080,"low":1.18031,"close":1.18046,"Volume":1166}
  ]}' | python3 -m json.tool

# ── Error: invalid OHLC (high < close) — 422 ─────────────────────────────────
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"candles": [
    {"time":"2026-02-26T19:15:00+05:30","open":1.18032,"high":1.17000,"low":1.18031,"close":1.18046,"Volume":1166}
  ]}' | python3 -m json.tool

# ── Error: negative price — 422 ───────────────────────────────────────────────
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"candles": [
    {"time":"2026-02-26T19:15:00+05:30","open":-1.18032,"high":1.18080,"low":1.18031,"close":1.18046,"Volume":1166}
  ]}' | python3 -m json.tool


# ══════════════════════════════════════════════════════════════════════════════
#  POST /predict/csv
#  Single prediction from an uploaded CSV file
#  File must have columns: time, open, high, low, close, Volume
# ══════════════════════════════════════════════════════════════════════════════

# ── Happy path — upload raw CSV ────────────────────────────────────────────────
curl -s -X POST http://localhost:8000/predict/csv \
  -F "file=@data/raw/OANDA_EURUSD_15.csv;type=text/csv" \
  | python3 -m json.tool

# ── Upload a custom CSV slice ──────────────────────────────────────────────────
#
# First export a slice:
#   python3 -c "
#   import pandas as pd
#   pd.read_csv('data/raw/OANDA_EURUSD_15.csv').tail(125).to_csv('context.csv', index=False)
#   print('context.csv created')
#   "
#
curl -s -X POST http://localhost:8000/predict/csv \
  -F "file=@context.csv;type=text/csv" \
  | python3 -m json.tool


# ── Expected response ─────────────────────────────────────────────────────────
# {
#   "colour":      "RED",
#   "green_prob":  0.486710,
#   "red_prob":    0.513290,
#   "confidence":  0.513290,
#   "signal":      0,
#   "last_candle": "2026-02-26T19:15:00+05:30",
#   "latency_ms":  301.7
# }

# ── Error: CSV with too few rows — 422 ───────────────────────────────────────
#   python3 -c "
#   import pandas as pd
#   pd.read_csv('data/raw/OANDA_EURUSD_15.csv').tail(5).to_csv('tiny.csv', index=False)
#   "
curl -s -X POST http://localhost:8000/predict/csv \
  -F "file=@tiny.csv;type=text/csv" \
  | python3 -m json.tool

# ── Error: corrupt file — 422/500 ─────────────────────────────────────────────
curl -s -X POST http://localhost:8000/predict/csv \
  -F "file=@/dev/null;type=text/csv" \
  | python3 -m json.tool


# ══════════════════════════════════════════════════════════════════════════════
#  POST /predict/batch
#  Predict every usable row in the payload (for backtesting)
#  Same JSON format as /predict — returns an array of predictions
# ══════════════════════════════════════════════════════════════════════════════

curl -s -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d @candles.json | python3 -m json.tool

# ── Pretty-print only the first 3 predictions ────────────────────────────────
curl -s -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d @candles.json \
  | python3 -c "
import sys, json
body = json.load(sys.stdin)
print(f\"n_predictions : {body['n_predictions']}\")
print(f\"latency_ms    : {body['latency_ms']}\")
print()
for p in body['predictions'][:3]:
    print(json.dumps(p, indent=2))
"


# ── Expected response ─────────────────────────────────────────────────────────
# {
#   "n_predictions": 29,
#   "latency_ms":    318.5,
#   "predictions": [
#     {
#       "candle_time": "2026-02-25T12:15:00+05:30",
#       "signal":      0,
#       "colour":      "RED",
#       "green_prob":  0.482311,
#       "red_prob":    0.517689,
#       "confidence":  0.517689
#     },
#     ...
#   ]
# }

# ── Count GREEN vs RED in the batch ──────────────────────────────────────────
curl -s -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d @candles.json \
  | python3 -c "
import sys, json
from collections import Counter
body   = json.load(sys.stdin)
counts = Counter(p['colour'] for p in body['predictions'])
total  = body['n_predictions']
print(f\"GREEN : {counts['GREEN']:>4}  ({counts['GREEN']/total*100:.1f}%)\")
print(f\"RED   : {counts['RED']:>4}  ({counts['RED']/total*100:.1f}%)\")
print(f\"Total : {total}\")
"


# ══════════════════════════════════════════════════════════════════════════════
#  POST /predict/batch/csv
#  Batch predictions from an uploaded CSV (full backtest in one call)
# ══════════════════════════════════════════════════════════════════════════════

curl -s -X POST http://localhost:8000/predict/batch/csv \
  -F "file=@data/raw/OANDA_EURUSD_15.csv;type=text/csv" \
  | python3 -m json.tool

# ── Summary stats from the full dataset ──────────────────────────────────────
curl -s -X POST http://localhost:8000/predict/batch/csv \
  -F "file=@data/raw/OANDA_EURUSD_15.csv;type=text/csv" \
  | python3 -c "
import sys, json
from collections import Counter
body   = json.load(sys.stdin)
preds  = body['predictions']
total  = body['n_predictions']
counts = Counter(p['colour'] for p in preds)
avg_conf = sum(p['confidence'] for p in preds) / total
print(f\"Total predictions : {total:,}\")
print(f\"GREEN             : {counts['GREEN']:,}  ({counts['GREEN']/total*100:.1f}%)\")
print(f\"RED               : {counts['RED']:,}  ({counts['RED']/total*100:.1f}%)\")
print(f\"Avg confidence    : {avg_conf:.4f}\")
print(f\"Latency           : {body['latency_ms']:.1f} ms\")
"


# ── Expected summary output ───────────────────────────────────────────────────
# Total predictions : 20,058
# GREEN             : 9,982  (49.8%)
# RED               : 10,076 (50.2%)
# Avg confidence    : 0.5134
# Latency           : 4821.3 ms


# ══════════════════════════════════════════════════════════════════════════════
#  ERROR CASES  —  expected HTTP status codes
# ══════════════════════════════════════════════════════════════════════════════

# 404 — unknown route
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/nonexistent
# → 404

# 405 — wrong method (/health is GET only)
curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8000/health
# → 405

# 422 — missing request body on /predict
curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" -d '{}'
# → 422

# 422 — tail below minimum on /predict/latest
curl -s -o /dev/null -w "%{http_code}" \
  "http://localhost:8000/predict/latest?tail=10"
# → 422


# ══════════════════════════════════════════════════════════════════════════════
#  POWER COMBOS
# ══════════════════════════════════════════════════════════════════════════════

# ── Poll prediction every 15 minutes (cron-style) ─────────────────────────────
watch -n 900 'curl -s "http://localhost:8000/predict/latest" | python3 -m json.tool'

# ── Save batch results to CSV for offline analysis ────────────────────────────
curl -s -X POST http://localhost:8000/predict/batch/csv \
  -F "file=@data/raw/OANDA_EURUSD_15.csv;type=text/csv" \
  | python3 -c "
import sys, json, csv
body = json.load(sys.stdin)
keys = ['candle_time','signal','colour','green_prob','red_prob','confidence']
writer = csv.DictWriter(sys.stdout, fieldnames=keys, extrasaction='ignore')
writer.writeheader()
writer.writerows(body['predictions'])
" > predictions.csv
echo "Saved to predictions.csv"

# ── Check API is live before sending data ─────────────────────────────────────
curl -sf http://localhost:8000/health \
  | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d['status']=='ok' else 1)" \
  && echo "API is ready" \
  || echo "API is not ready — run: python run.py serve"

# ── One-liner: generate candles.json from raw CSV, then predict ───────────────
python3 -c "
import pandas as pd, json
df = pd.read_csv('data/raw/OANDA_EURUSD_15.csv')
json.dump({'candles': df.tail(125).to_dict(orient='records')}, open('candles.json','w'))
print('candles.json ready')
" && curl -s -X POST http://localhost:8000/predict \
       -H "Content-Type: application/json" \
       -d @candles.json | python3 -m json.tool