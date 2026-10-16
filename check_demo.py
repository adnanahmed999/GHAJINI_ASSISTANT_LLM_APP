"""
Run this before the demo: checks that RAG finds the right document for every demo question.

  python check_demo.py

Each question must find its document in the top 3. If one shows ❌, the document text
needs more words that match the question's meaning (fix it in make_docs.py, then
run make_docs.py and rag.py --rebuild again).
"""

import rag

# question -> the start of the title of the document that must be found
DEMO = {
    "Am I allergic to anything?": "Doctor visit – Dr. Mehta",
    "Can I eat at the new Thai restaurant?": "Doctor visit – Dr. Mehta",
    "What's my Wi-Fi password?": "Home internet setup",
    "Is my passport still valid?": "Passport",
    "When does my bike insurance end?": "Vehicle policy renewed",
    "Is my laptop still under warranty?": "Laptop bill",
    "What should I gift Kalpana for her birthday?": "Kalpana – health update",
    "Where are my bank locker keys?": "Bank locker",
    "What's my blood group?": "Blood test report",
    "Who is my landlord?": "Flat payment",
}

TOP = 3
failed = 0
for question, expected in DEMO.items():
    hits = rag.search(question, k=10)
    rank = next((i for i, h in enumerate(hits, 1) if h["title"].startswith(expected)), None)
    ok = rank is not None and rank <= TOP
    failed += not ok
    where = f"rank {rank}" if rank else "not in top 10"
    print(f"{'✅' if ok else '❌'} {question:<46} -> {expected} ({where}, distance {hits[(rank or 1) - 1]['distance']})")

print(f"\n{len(DEMO) - failed}/{len(DEMO)} demo questions find their document in the top {TOP}.")
if failed:
    print("Tell Claude which ones failed, so the document text can be improved.")
