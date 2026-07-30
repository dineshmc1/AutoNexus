"""Environment-driven optional service configuration."""

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

LLM_MODEL = os.getenv("LLM_MODEL")
