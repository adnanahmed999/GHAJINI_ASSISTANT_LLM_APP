"""
MCP SERVER: a small "Calendar and Reminders" service.

In real life another team or company would build and run this. Our agent only
connects to it as an MCP client and discovers its tools by itself.

Tools:
  get_today_date       -> the LLM has no clock, so it asks this
  upcoming_deadlines   -> documents whose expiry / renewal / due date is coming up (no AI, just a date check)

The agent starts this file automatically. To try it alone:  python mcp_server.py
"""

from datetime import date, timedelta

from mcp.server.fastmcp import FastMCP

from config import DEMO_DATE
from doc_store import doc_to_text, load_docs

mcp = FastMCP("Calendar and Reminders")


def _today() -> date:
    return date.fromisoformat(DEMO_DATE) if DEMO_DATE else date.today()


@mcp.tool()
def get_today_date() -> str:
    """Get today's date and day of the week. Use this whenever the answer depends on what day it is
    (is something still valid, how many days are left, has it expired)."""
    t = _today()
    return f"Today is {t.strftime('%A, %d %B %Y')} ({t.isoformat()})."


@mcp.tool()
def upcoming_deadlines(days: int = 30) -> str:
    """List Sanjay's documents whose due date, expiry or renewal falls within the next `days` days.
    Use this for questions like "what do I need to renew or pay soon?"."""
    today = _today()
    until = today + timedelta(days=days)
    found = sorted(
        (d for d in load_docs() if d.get("due") and today.isoformat() <= d["due"] <= until.isoformat()),
        key=lambda d: d["due"],
    )
    if not found:
        return f"Nothing is due between {today:%d %b %Y} and {until:%d %b %Y}."
    head = f"{len(found)} document(s) due between {today:%d %b %Y} and {until:%d %b %Y}:"
    return head + "\n\n" + "\n\n".join(doc_to_text(d) for d in found)


if __name__ == "__main__":
    mcp.run()  # talks MCP over stdin/stdout
