"""Settings shared by all scripts. Values come from the .env file."""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DOCS_DIR = BASE_DIR / "data" / "docs"                      # one JSON file per document
EMBEDDINGS_FILE = BASE_DIR / "data" / "embeddings.json"    # saved vector store (created on first run)

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini")

# Pretend "today" is this date during the demo (format YYYY-MM-DD).
# Leave empty to use the real date.
DEMO_DATE = os.getenv("DEMO_DATE", "")
