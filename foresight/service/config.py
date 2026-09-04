"""
service/config.py — Centralized environment and path settings for Project FORESIGHT.
"""

import os
from pathlib import Path

# Try importing python-dotenv if available
try:
    from dotenv import load_dotenv
    # Load .env from project root
    project_root = Path(__file__).resolve().parent.parent
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=env_file, override=True)
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Backend Host & Port Configuration
API_HOST = os.getenv("FORESIGHT_API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("FORESIGHT_API_PORT", "8000"))
API_URL = os.getenv("FORESIGHT_API_URL", f"http://{API_HOST}:{API_PORT}")
