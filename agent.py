"""
THE AGENT: Ghajini, Sanjay's memory assistant, built with LangGraph.

  ┌─────────┐  wants a tool?  ┌─────────┐
  │  agent  │ ──────────────► │  tools  │
  │  (LLM)  │ ◄────────────── │         │
  └─────────┘   tool result   └─────────┘
       │ no more tools needed
       ▼
     answer

Tools the agent can use:
  - search_documents     our own tool (RAG over Sanjay's documents)
  - get_today_date       from the MCP server
  - upcoming_deadlines   from the MCP server

Try it:  python agent.py "Am I allergic to anything?"
"""

import asyncio
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

import rag
from config import CHAT_MODEL

SYSTEM_PROMPT = """You are Ghajini, Sanjay's personal memory assistant. Sanjay has short-term memory loss,
so he saves a note or document for everything: medical, bills, IDs, passwords, warranties, people, travel, money.
You answer his questions using those documents.

Rules:
- Use search_documents to find information. The documents often use different words than the question
  (for example "router key" instead of "Wi-Fi password"), so think about what the note might say.
  If the first search does not answer the question, search again with different words.
- You do not know today's date. Call get_today_date whenever the answer depends on the date
  (is something still valid, has it expired, how many days are left).
- For questions about what is due, expiring or needs renewing or paying soon, use upcoming_deadlines.
- If two documents disagree, trust the one saved more recently.
- Always mention the title and date of the documents you used.
- If you cannot find something in the documents, say so. Never make up facts.
- For small talk like "hi" or "thanks", just reply. Do not search.
- Keep answers short and clear. If Sanjay needs to do something, end with one line starting "To do:"."""


@tool
def search_documents(query: str) -> str:
    """Search Sanjay's saved notes and documents by meaning. Returns the 5 most relevant documents with their dates."""
    hits = rag.search(query, k=5)
    return "\n\n".join(f"(distance {h['distance']}, lower = closer)\n{h['text']}" for h in hits)


def build_graph(llm, tools):
    """The agent loop: the LLM thinks, calls tools if needed, reads the results, and repeats."""
    llm_with_tools = llm.bind_tools(tools)

    async def agent_node(state: MessagesState):
        reply = await llm_with_tools.ainvoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])
        return {"messages": [reply]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)  # tool call? -> "tools", else -> end
    graph.add_edge("tools", "agent")
    return graph.compile()


# The MCP server runs as a separate program; we connect to it as an MCP client.
MCP_SERVERS = {
    "calendar": {
        "command": sys.executable,
        "args": [str(Path(__file__).with_name("mcp_server.py"))],
        "transport": "stdio",
    }
}

_graph = None


async def get_graph(llm=None):
    global _graph
    if _graph is None or llm is not None:
        mcp_tools = await MultiServerMCPClient(MCP_SERVERS).get_tools()   # discover the server's tools
        tools = [search_documents, *mcp_tools]
        _graph = build_graph(llm or ChatOpenAI(model=CHAT_MODEL), tools)
    return _graph


async def ask(question: str, history: list | None = None, llm=None) -> dict:
    """Ask the agent a question.

    Returns {"answer": str, "steps": [{"tool", "args", "result"}], "messages": [...]}.
    Pass the returned "messages" back in as history to keep the conversation going.
    """
    graph = await get_graph(llm)
    messages = list(history or []) + [HumanMessage(question)]
    result = await graph.ainvoke({"messages": messages}, {"recursion_limit": 15})

    new_messages = result["messages"][len(messages):]
    steps, calls = [], {}
    for m in new_messages:
        if isinstance(m, AIMessage):
            for c in m.tool_calls:
                calls[c["id"]] = c
        elif isinstance(m, ToolMessage):
            c = calls.get(m.tool_call_id, {})
            if isinstance(m.content, str):
                content = m.content
            else:  # MCP tools return a list of content blocks
                content = "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in m.content)
            steps.append({"tool": c.get("name", m.name), "args": c.get("args", {}), "result": content})

    return {"answer": result["messages"][-1].content, "steps": steps, "messages": result["messages"]}


async def _main() -> None:
    question = " ".join(sys.argv[1:]) or "Am I allergic to anything?"
    print(f"\nSanjay: {question}\n")
    out = await ask(question)
    for i, s in enumerate(out["steps"], 1):
        preview = s["result"].replace("\n", " ")[:120]
        print(f"  step {i}: {s['tool']}({s['args']})\n          -> {preview}...")
    print(f"\nGhajini: {out['answer']}\n")


if __name__ == "__main__":
    asyncio.run(_main())
