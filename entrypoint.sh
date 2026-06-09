#!/bin/bash
set -e
python download_models.py
exec uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
