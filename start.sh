#!/usr/bin/env bash
set -e

echo "==============================================="
echo " Starting LVMH Financial Intelligence (ALL-IN-ONE)"
echo " API:      http://127.0.0.1:8000"
echo " Streamlit: http://127.0.0.1:${PORT:-7860}"
echo "==============================================="

# Lancer FastAPI en background
uvicorn api.app:app --host 0.0.0.0 --port 8000 &

# Lancer Streamlit au port Spaces
streamlit run ui/app.py \
  --server.port "${PORT:-7860}" \
  --server.address "0.0.0.0" \
  --server.headless true \
  --browser.gatherUsageStats false
