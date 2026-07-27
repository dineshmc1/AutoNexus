import os
from dotenv import load_dotenv

# Load API keys from .env file (e.g., OPENROUTER_API_KEY)
load_dotenv()

USE_WANDB = True
WANDB_PROJECT = "automl-meta-learning"
WANDB_ENTITY = None  # Add entity here if required

USE_LLM = True             # set False to skip API calls (faster)
LLM_MODEL = os.getenv("LLM_MODEL", "openrouter/meta-llama/llama-3.3-70b-instruct:free")
