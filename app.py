"""
The Ghajini LLM App: a ChatGPT-style UI for Ghajini, Sanjay's memory assistant over his saved documents.

  - Left:   past chats (kept until you refresh the page) and "New chat"
  - Middle: the conversation
  - Right:  click "🧠 Show thinking" on any answer to see every step the agent took

  streamlit run app.py
"""

import asyncio
import html
import re
import threading
import time
import uuid
from pathlib import Path

import streamlit as st
from streamlit.errors import StreamlitAPIException

from agent import ask
from config import CHAT_MODEL, DEMO_DATE

APP_NAME = "The Ghajini LLM App"
ASSETS = Path(__file__).resolve().parent / "assets"

st.set_page_config(page_title=APP_NAME, page_icon="🧠", layout="wide")


# ---------------------------------------------------------------------------
# Ghajini's mood pictures: assets/ghajini_lost.* and assets/ghajini_confident.*
# (if a picture is missing, an emoji is shown instead)
# ---------------------------------------------------------------------------
def find_image(name: str) -> str | None:
    for ext in ("png", "jpg", "jpeg", "webp", "gif"):
        p = ASSETS / f"{name}.{ext}"
        if p.exists():
            return str(p)
    return None


MOODS = {
    "lost": {
        "image": find_image("ghajini_lost"),
        "emoji": "😵",
        "text": "Main kaun hoon? Main kahaan hoon? Maine yeh kahan likha tha?? 😵",
    },
    "confident": {
        "image": find_image("ghajini_confident"),
        "emoji": "😎",
        "text": "Yaad aa gaya! 😎",
    },
}


# ---------------------------------------------------------------------------
# What each tool is, for the thinking panel
# ---------------------------------------------------------------------------
TOOLS = {
    "search_documents": {
        "icon": "🔎",
        "short": "RAG search",
        "kind": "RAG",
        "source": "Our own tool · RAG over Sanjay's documents",
        "explain": "The search words were turned into 1536 numbers (an embedding). FAISS compared them with the "
                   "numbers of every saved document and returned the closest ones. Lower distance = closer in meaning. "
                   "Only these few documents are sent to the LLM, not the whole folder.",
    },
    "get_today_date": {
        "icon": "📅",
        "short": "Today's date",
        "kind": "MCP",
        "source": "MCP server · Calendar and Reminders",
        "explain": "The LLM has no clock, so it asked an outside service for today's date. "
                   "The agent found this tool by itself when it connected to the MCP server.",
    },
    "upcoming_deadlines": {
        "icon": "⏰",
        "short": "Upcoming deadlines",
        "kind": "MCP",
        "source": "MCP server · Calendar and Reminders",
        "explain": "A plain date check on the MCP server: it read the due / expiry date of every document and "
                   "listed the ones coming up. No AI and no embeddings needed.",
    },
}
DEFAULT_TOOL = {"icon": "🛠️", "short": "Tool", "kind": "Tool", "source": "Tool", "explain": ""}


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
.doc { border-left:3px solid rgba(59,130,246,0.6); background:rgba(59,130,246,0.07);
       padding:8px 11px; border-radius:6px; margin:4px 0 12px; font-size:0.88rem; line-height:1.4; }
.doc .meta { display:block; font-size:0.75rem; opacity:0.7; margin-bottom:4px; }
.doc .due { display:inline-block; font-size:0.72rem; padding:1px 7px; border-radius:999px;
            background:rgba(234,179,8,0.2); margin-left:6px; }
.flow { display:flex; flex-wrap:wrap; align-items:center; gap:6px; margin:2px 0 10px; font-size:0.8rem; }
.chip { padding:3px 9px; border-radius:999px; border:1px solid rgba(128,128,128,0.35); white-space:nowrap; }
.chip.rag { background:rgba(59,130,246,0.14); }
.chip.mcp { background:rgba(168,85,247,0.14); }
.chip.llm { background:rgba(234,179,8,0.16); }
.arrow { opacity:0.5; }
.doclabel { font-size:0.85rem; font-weight:600; margin-top:6px; }
.mood { font-size:1.05rem; font-style:italic; opacity:0.85; }
.bigemoji { font-size:7rem; text-align:center; line-height:1.1; }
.smallemoji { font-size:3.5rem; line-height:1; }
/* narrower sidebar (change 240px to make it wider or narrower) */
section[data-testid="stSidebar"]:not([data-collapsed="true"]):not([aria-expanded="false"]) { width:240px !important; min-width:240px !important; max-width:240px !important; }
</style>
""",
    unsafe_allow_html=True,
)


def _try(fn, *args, options: list[dict], **kwargs):
    """Call a Streamlit function with the first set of options this Streamlit version accepts."""
    for extra in options:
        try:
            return fn(*args, **extra, **kwargs)
        except (TypeError, StreamlitAPIException):
            continue
    return fn(*args, **kwargs)


def stretch_button(label: str, **kwargs) -> bool:
    """A full-width button that works on old and new Streamlit versions."""
    return _try(st.button, label, options=[{"width": "stretch"}, {"use_container_width": True}], **kwargs)


# ---------------------------------------------------------------------------
# Run the async agent from Streamlit (one background event loop for the app)
# ---------------------------------------------------------------------------
@st.cache_resource
def _event_loop():
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    return loop


def run_agent(question: str, history: list) -> dict:
    return asyncio.run_coroutine_threadsafe(ask(question, history=history), _event_loop()).result()


# ---------------------------------------------------------------------------
# Chats live in session_state, so they last until the page is refreshed
# ---------------------------------------------------------------------------
def new_chat() -> None:
    current = st.session_state.get("current")
    if current and not st.session_state.chats[current]["turns"]:
        return  # already on an empty chat
    cid = uuid.uuid4().hex[:8]
    st.session_state.chats[cid] = {"title": "New chat", "turns": [], "history": []}
    st.session_state.order.append(cid)
    st.session_state.current = cid
    st.session_state.thinking = None


if "chats" not in st.session_state:
    st.session_state.chats = {}
    st.session_state.order = []
    st.session_state.current = None
    new_chat()

chat = st.session_state.chats[st.session_state.current]


def mood_of(c: dict) -> str:
    return "confident" if c["turns"] and not c["turns"][-1].get("error") else "lost"


# ---------------------------------------------------------------------------
# Turn tool results (plain text) back into documents for display
# ---------------------------------------------------------------------------
DIST_RE = re.compile(r"^\(distance ([\d.]+)")
FIELDS = {"Title: ": "title", "Type: ": "type", "Saved on: ": "date", "Due / valid till: ": "due"}


def parse_docs(text: str) -> tuple[list[dict], list[str]]:
    """Split a tool result into documents. Returns (documents, other lines such as a heading)."""
    docs, other = [], []
    for block in text.split("\n\n"):
        lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
        doc = {"distance": None, "title": None, "type": "", "date": "", "due": "", "body": []}
        for line in lines:
            m = DIST_RE.match(line)
            if m:
                doc["distance"] = float(m.group(1))
                continue
            field = next((f for f in FIELDS if line.startswith(f)), None)
            if field and not doc["body"]:
                doc[FIELDS[field]] = line[len(field):]
            else:
                doc["body"].append(line)
        if doc["title"]:
            docs.append(doc)
        else:
            other.extend(lines)
    return docs, other


def render_doc(doc: dict, label: str = "") -> None:
    head = f"{label}📄 {html.escape(doc['title'])}"
    if doc["distance"] is not None:
        head += f" &nbsp;·&nbsp; distance <code>{doc['distance']:.3f}</code>"
    meta = f"{html.escape(doc['type'])} · saved on {html.escape(doc['date'])}"
    if doc["due"]:
        meta += f"<span class='due'>⏰ due {html.escape(doc['due'])}</span>"
    body = html.escape(" ".join(doc["body"]))
    st.markdown(f"<div class='doclabel'>{head}</div><div class='doc'><span class='meta'>{meta}</span>{body}</div>",
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Thinking panel (right side)
# ---------------------------------------------------------------------------
def render_step(step: dict) -> None:
    info = TOOLS.get(step["tool"], DEFAULT_TOOL)
    st.markdown(f"**{info['icon']} `{step['tool']}`**  \n<small>{info['source']}</small>", unsafe_allow_html=True)
    if step["args"]:
        args = ", ".join(f"{k} = {v!r}" for k, v in step["args"].items())
        st.markdown(f"**Asked with:** `{args}`")
    else:
        st.markdown("**Asked with:** *(nothing)*")

    docs, other = parse_docs(step["result"])
    if docs:
        if step["tool"] == "search_documents":
            st.markdown(f"**Got back {len(docs)} documents**, closest first:")
            for rank, doc in enumerate(docs, 1):
                render_doc(doc, label=f"#{rank} &nbsp;")
        else:
            st.markdown(f"**Got back:** {' '.join(other)}" if other else f"**Got back {len(docs)} document(s):**")
            for doc in docs:
                render_doc(doc)
    else:
        st.markdown("**Got back:**")
        st.info(step["result"])

    if info["explain"]:
        st.caption("💡 " + info["explain"])


def flow_nodes(turn: dict) -> list[dict]:
    """The agent loop as a list of nodes: Question -> LLM -> tool -> LLM -> ... -> Answer."""
    steps = turn["steps"]
    nodes = [{"kind": "question", "label": "💬 Question"}]
    for i, step in enumerate(steps):
        info = TOOLS.get(step["tool"], DEFAULT_TOOL)
        nodes.append({"kind": "llm_call", "label": "🧠 LLM decides", "step": step})
        nodes.append({"kind": "tool", "label": f"{info['icon']} {info['short']} · {info['kind']}", "step": step})
    nodes.append({"kind": "llm_answer", "label": "🧠 LLM writes the answer" if steps else "🧠 LLM answers directly"})
    nodes.append({"kind": "answer", "label": "✅ Answer"})
    return nodes


def render_node_details(node: dict, turn: dict) -> None:
    kind = node["kind"]
    if kind == "question":
        st.markdown(f"> {turn['question']}")
        st.caption("💡 The question goes to the agent together with the earlier messages of this chat (memory).")
    elif kind == "llm_call":
        step = node["step"]
        args = ", ".join(f"{k}={v!r}" for k, v in step["args"].items())
        st.markdown(f"The LLM decided it needs a tool and asked for:\n\n`{step['tool']}({args})`")
        st.caption("💡 The LLM only *asks* for a tool. Our code runs the tool and gives the result back.")
    elif kind == "tool":
        render_step(node["step"])
    elif kind == "llm_answer":
        if turn["steps"]:
            st.markdown("The LLM read the question plus every tool result, decided it had enough, and wrote the answer.")
        else:
            st.markdown("The LLM decided no tool was needed (small talk) and answered straight away.")
        st.caption("💡 Its rules: check the date, trust newer messages, always give dates, never invent facts.")
    elif kind == "answer":
        steps = turn["steps"]
        rag_calls = sum(1 for s in steps if s["tool"] == "search_documents")
        mcp_calls = sum(1 for s in steps if TOOLS.get(s["tool"], DEFAULT_TOOL)["kind"] == "MCP")
        st.markdown(
            f"- ⏱️ Time: **{turn['seconds']:.1f}s**\n"
            f"- 🔎 RAG searches: **{rag_calls}**\n"
            f"- 🔌 MCP calls: **{mcp_calls}**\n"
            f"- 🧠 LLM: `{CHAT_MODEL}`"
        )


def render_thinking(turn: dict) -> None:
    top = st.columns([5, 1])
    top[0].markdown("#### 🧠 Thinking")
    if top[1].button("✖", key="close_thinking", help="Close"):
        st.session_state.thinking = None
        st.rerun()
    st.caption(f"Question: {turn['question']}  \nClick a step to see what happened.")

    nodes = flow_nodes(turn)
    selected = st.session_state.get("node")
    for i, node in enumerate(nodes):
        if i > 0:
            st.markdown("<div style='text-align:center; opacity:0.5; margin:-6px 0;'>↓</div>", unsafe_allow_html=True)
        is_open = selected == i
        if stretch_button(node["label"], key=f"node_{i}", type="primary" if is_open else "secondary"):
            st.session_state.node = None if is_open else i
            st.rerun()
        if is_open:
            with st.container(border=True):
                render_node_details(node, turn)


# ---------------------------------------------------------------------------
# Chat area (middle)
# ---------------------------------------------------------------------------
HEADER_IMAGE_WIDTH = 170   # same size for the lost and the confident picture (pixels)


@st.cache_data
def image_data_uri(path: str) -> str:
    """Read a picture once and turn it into text that HTML can show directly."""
    import base64
    import mimetypes
    mime = mimetypes.guess_type(path)[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


def show_mood_header(mood: str) -> None:
    """Same header in every state: Ghajini's picture (lost or confident), the app name and his mood.

    Built as one HTML flex row, so the text always sits next to the picture and never
    on top of it, even when the thinking panel makes the chat column narrow.
    """
    m = MOODS[mood]
    if m["image"]:
        pic = (f"<img src='{image_data_uri(m['image'])}' alt='Ghajini {mood}' "
               f"style='width:{HEADER_IMAGE_WIDTH}px; max-width:38%; height:auto; border-radius:10px; flex-shrink:0;'>")
    else:
        pic = f"<div class='smallemoji'>{m['emoji']}</div>"
    header_html = (
        f"""<div style='display:flex; align-items:center; gap:18px; margin-bottom:10px;'>
              {pic}
              <div style='min-width:0;'>
                <div style='font-size:2rem; font-weight:700; line-height:1.2;'>{APP_NAME}</div>
                <div class='mood' style='margin-top:4px;'>{m['text']}</div>
              </div>
            </div>"""
    )
    # one line, so Markdown never mistakes the indented HTML for a code block
    st.markdown(" ".join(header_html.split()), unsafe_allow_html=True)


def show_empty_state() -> None:
    show_mood_header("lost")
    st.caption("Sanjay forgets everything, so he saves a note for everything. "
               "Ask me anything, I'll search his documents by meaning.")


def assistant_avatar(ok: bool):
    m = MOODS["confident" if ok else "lost"]
    return m["image"] or m["emoji"]


def render_turn(i: int, turn: dict) -> None:
    with st.chat_message("user", avatar="💬"):
        st.markdown(turn["question"])

    with st.chat_message("assistant", avatar=assistant_avatar(not turn.get("error"))):
        if turn.get("error"):
            st.error(f"Ghajini forgot again... something went wrong: {turn['error']}")
            return
        st.markdown(turn["answer"])
        n = len(turn["steps"])
        is_open = st.session_state.thinking == i
        label = "🧠 Hide thinking" if is_open else f"🧠 Show thinking · {n} step{'s' if n != 1 else ''}"
        if st.button(label, key=f"think_{st.session_state.current}_{i}", type="primary" if is_open else "secondary"):
            st.session_state.thinking = None if is_open else i
            st.session_state.node = None
            st.rerun()


# ---------------------------------------------------------------------------
# Sidebar: ChatGPT-style list of chats
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### 🧠 {APP_NAME}")
    if stretch_button("➕ New chat"):
        new_chat()
        st.rerun()

    st.caption("Chats")
    for cid in reversed(st.session_state.order):
        c = st.session_state.chats[cid]
        if not c["turns"] and cid != st.session_state.current:
            continue
        title = c["title"] if len(c["title"]) <= 26 else c["title"][:24] + "…"
        if stretch_button(title, key=f"chat_{cid}",
                          type="primary" if cid == st.session_state.current else "secondary"):
            st.session_state.current = cid
            st.session_state.thinking = None
            st.rerun()


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
prompt = st.chat_input("Ask Ghajini anything Sanjay saved...")

thinking_turn = None
if st.session_state.thinking is not None and st.session_state.thinking < len(chat["turns"]):
    thinking_turn = chat["turns"][st.session_state.thinking]

if thinking_turn:
    chat_col, think_col = st.columns([3, 2], gap="large")
else:
    chat_col, think_col = st.container(), None

with chat_col:
    if chat["turns"] or prompt:
        show_mood_header(mood_of(chat) if chat["turns"] else "lost")
    else:
        show_empty_state()

    for i, turn in enumerate(chat["turns"]):
        render_turn(i, turn)

    if prompt:
        with st.chat_message("user", avatar="💬"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar=assistant_avatar(False)):
            with st.spinner("Searching Sanjay's documents... 🧠"):
                start = time.perf_counter()
                try:
                    out = run_agent(prompt, chat["history"])
                    turn = {"question": prompt, "answer": out["answer"], "steps": out["steps"]}
                    chat["history"] = out["messages"]
                except Exception as e:  # show the error in the chat instead of crashing
                    turn = {"question": prompt, "error": str(e), "steps": []}
                turn["seconds"] = time.perf_counter() - start
        if not chat["turns"]:
            chat["title"] = prompt
        chat["turns"].append(turn)
        st.rerun()

if think_col is not None:
    with think_col:
        with st.container(border=True, height=720):
            render_thinking(thinking_turn)
