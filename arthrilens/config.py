import os
from dotenv import load_dotenv

# Find root directory (where .env file resides)
# config.py is in project_root/arthrilens/, so root is one level up
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
env_path = os.path.join(root_dir, ".env")

# Load environment variables
load_dotenv(env_path)

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Default Models
EMBEDDING_MODEL = "gemini-embedding-2"
GEMINI_MODEL = "gemini-3.5-flash"
GROQ_MODEL = "groq/compound"
OPENROUTER_MODEL = "openrouter/free"

# Paths
DEFAULT_DATA_DIR = os.path.join(script_dir, "data")
DEFAULT_VECTOR_STORE_PATH = os.path.join(DEFAULT_DATA_DIR, "vector_store.pkl")
