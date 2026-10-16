"""
Slide 1 demo: plain Ctrl+F (keyword) search over Sanjay's documents.

  python ctrl_f.py allergy      -> 0 results, even though the peanut note is right there
  python ctrl_f.py peanut       -> found, but only if you already know the exact word

Compare with RAG (search by meaning):
  python rag.py "am I allergic to anything?"
"""

import sys

from doc_store import doc_to_text, load_docs

word = " ".join(sys.argv[1:]) or "allergy"
hits = [d for d in load_docs() if word.lower() in doc_to_text(d).lower()]

print(f'\nCtrl+F "{word}" in {len(load_docs())} documents -> {len(hits)} result(s)\n')
for d in hits:
    print(f"  - {d['date']}  {d['title']}")
if not hits:
    print("  Nothing found. 😵")
