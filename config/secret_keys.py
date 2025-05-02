"""
Load environment variables from a .env file
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Get the parent directory of AIEngine
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Construct path to .env file
ENV_PATH = BASE_DIR / '.env'

load_dotenv(ENV_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file in the parent directory.")
