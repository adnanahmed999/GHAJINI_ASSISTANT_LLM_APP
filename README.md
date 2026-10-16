# The Ghajini LLM App

Sanjay forgets everything (like Ghajini), so he saves a note for everything: doctor visits, bills,
IDs, passwords, warranties, people, travel. Ghajini is his assistant: a LangGraph agent that searches
those documents **by meaning** (RAG), asks an MCP server for today's date and upcoming deadlines,
and answers his questions.

## Files

| File | What it is |
|---|---|
| `make_docs.py` | Creates Sanjay's documents (one JSON file per document in `data/docs/`) |
| `doc_store.py` | Reads the documents and turns each one into text (one document = one chunk) |
| `ctrl_f.py` | Slide 1 demo: plain keyword search, to show where Ctrl+F fails |
| `rag.py` | RAG: OpenAI embeds the documents once into `data/embeddings.json`; FAISS finds the closest ones. `python rag.py --rebuild` re-creates the JSON. |
| `check_demo.py` | Checks that RAG finds the right document for every demo question (run before the demo) |
| `mcp_server.py` | MCP server with `get_today_date` and `upcoming_deadlines` |
| `agent.py` | LangGraph agent: LLM + `search_documents` tool + MCP tools |
| `app.py` | Streamlit UI (ChatGPT-style): past chats on the left, "🧠 Show thinking" flow chart on the right |
| `assets/` | `ghajini_lost.png` (new chat) and `ghajini_confident.png` (after a good answer) |
| `check_connection.py` | Quick test that your OpenAI key and network work |
| `config.py` | Settings |

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate             # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env             # then put your OpenAI key in .env

python make_docs.py                # already done: data/docs/ is included
python rag.py --rebuild            # embed the documents (once)
python check_demo.py               # every demo question should show ✅
streamlit run app.py
```

## Demo questions

| Ask | What it shows |
|---|---|
| Am I allergic to anything? | The note never says "allergy" (Ctrl+F finds nothing), RAG finds the peanut note by meaning |
| What's my Wi-Fi password? | The note says "router key: tiger@123" |
| Is my passport still valid? | RAG finds "valid till 14 Nov 2026", the date tool (MCP) says today is 17 Oct 2026 → renew soon |
| What do I need to renew or pay soon? | `upcoming_deadlines` (MCP): plain date check, no AI |
| What should I gift Kalpana for her birthday? | Two notes disagree (loves cake vs stopped sugar); the newer one wins |
| hi | Small talk: no tools |

`DEMO_DATE` in `.env` makes "today" 17 Oct 2026 so the date answers work on any day.
