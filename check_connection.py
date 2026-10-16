"""
Step 0: check that your OpenAI key and network work, before running RAG or the agent.

  python check_connection.py

Makes two tiny calls (costs a tiny fraction of a cent):
  1. a normal chat message to the LLM
  2. one embedding (what RAG uses)
"""

from openai import OpenAI

from config import CHAT_MODEL, EMBED_MODEL

client = OpenAI()  # reads OPENAI_API_KEY from your environment or .env

print(f"1. Chat model ({CHAT_MODEL})...")
reply = client.chat.completions.create(
    model=CHAT_MODEL,
    messages=[{"role": "user", "content": "Say hello to Sanjay in five words."}],
)
print("   OK ->", reply.choices[0].message.content)

print(f"2. Embedding model ({EMBED_MODEL})...")
emb = client.embeddings.create(model=EMBED_MODEL, input="Router key: tiger@123")
print(f"   OK -> got {len(emb.data[0].embedding)} numbers for one sentence")

print("\nAll good. Next: python rag.py")
