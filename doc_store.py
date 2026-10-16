"""Reads Sanjay's documents and turns one document into plain text."""

import json
from datetime import date

from config import DOCS_DIR


def load_docs() -> list[dict]:
    """All documents, oldest first.
    Each item: {"id", "date": "YYYY-MM-DD", "type", "title", "text", "due" (optional)}."""
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in DOCS_DIR.glob("*.json")]
    return sorted(docs, key=lambda d: (d["date"], d["id"]))


def nice_date(iso: str) -> str:
    return date.fromisoformat(iso).strftime("%d %B %Y")


def doc_to_text(doc: dict) -> str:
    """One document as readable text. This is what we embed (one document = one chunk) and what the LLM reads."""
    lines = [
        f"Title: {doc['title']}",
        f"Type: {doc['type']}",
        f"Saved on: {nice_date(doc['date'])}",
    ]
    if doc.get("due"):
        lines.append(f"Due / valid till: {nice_date(doc['due'])}")
    lines.append(doc["text"])
    return "\n".join(lines)
