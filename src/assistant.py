"""
Personal Assistant Agent — a minimal LangGraph implementation
=============================================================
A small, self-contained demo that accompanies the research survey
"Personal Assistant Agents: A Research Survey on Architectures, Privacy,
and Evaluation".

It illustrates three ideas from the survey:
  1. Architecture  — a supervisor routes a request to specialist sub-agents
                     (email / calendar) that run ReAct tool loops.
  2. Privacy/safety — destructive actions (send email, create event) pass
                     through a human-in-the-loop (HITL) gate before firing.
  3. Statefulness  — the graph checkpoints state so it can pause at the gate
                     and resume after a human decision.

The tools here are in-memory stubs, so the demo runs with no external
services. The LLM is served by the University of Passau OpenAI-compatible
endpoint; set OPENAI_API_KEY (and optionally LLM_MODEL / OPENAI_BASE_URL)
in a .env file or the environment.

Run:
    python assistant.py            # run the two demo tasks
    python assistant.py --graph    # write architecture.html (graph diagram)
"""

from __future__ import annotations

import json
import os
import uuid
import sys
from typing import Annotated, Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root

try:  # load .env from the repo root if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass

LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3-next-80b-a3b-instruct")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://llms.innkube.fim.uni-passau.de")


# ─────────────────────────────────────────────────────────────
# Shared state — every node reads from and writes to this dict.
# ─────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # conversation history
    intent: str | None                       # email | calendar | cross_service
    pending_action: dict | None              # gated call awaiting approval
    hitl_decision: str | None                # approve | reject


# ─────────────────────────────────────────────────────────────
# In-memory "world" — stands in for Gmail / Calendar so the demo
# runs without any external service. Tools mutate these lists in
# place, so the UI can show the inbox / calendar / drafts changing
# as the agent works. reset_world() restores the seed data.
# ─────────────────────────────────────────────────────────────
import copy

_SEED_INBOX = [
    {"id": "m1", "from": "anna@acme.com", "date": "2026-06-26", "label": None,
     "subject": "1h project sync this week?",
     "body": "Hi! Could we grab an hour this week for a project sync? "
             "I'd like to walk through the roadmap. Any weekday morning works for me."},
    {"id": "m2", "from": "david@acme.com", "date": "2026-06-26", "label": None,
     "subject": "Re: Q3 budget review",
     "body": "Thanks for the draft. Can you confirm the marketing line item by Friday? "
             "Finance needs the final number before we submit."},
    {"id": "m3", "from": "news@pythonweekly.com", "date": "2026-06-25", "label": None,
     "subject": "Python Weekly #640",
     "body": "This week: structural pattern matching tips, a new async HTTP client, "
             "and 12 new open-source projects. Read online."},
    {"id": "m4", "from": "it-support@company.com", "date": "2026-06-25", "label": None,
     "subject": "Action required: your password expires in 3 days",
     "body": "Your account password will expire on 2026-06-28. Please reset it via the "
             "internal portal to avoid being locked out."},
    {"id": "m5", "from": "no-reply@lottery-winner.biz", "date": "2026-06-24", "label": None,
     "subject": "🎉 You WON $1,000,000 !!! CLAIM NOW",
     "body": "Congratulations!!! You are our lucky winner. Send your bank details "
             "within 24 hours to claim your prize. Hurry!!!"},
    {"id": "m6", "from": "notifications@linkedin.com", "date": "2026-06-24", "label": None,
     "subject": "You appeared in 7 searches this week",
     "body": "See who's been looking at your profile. Upgrade to Premium for full insights."},
    {"id": "m7", "from": "sara@acme.com", "date": "2026-06-26", "label": None,
     "subject": "Can you review my slides before Thursday?",
     "body": "Hey, I've finished the draft deck for the client pitch. Could you review it "
             "and send comments before our Thursday rehearsal? Thanks a lot!"},
    {"id": "m8", "from": "recruiter@talenthub.io", "date": "2026-06-25", "label": None,
     "subject": "Exciting Senior Engineer role — interested?",
     "body": "Hi, I came across your profile and have a great opportunity with a top "
             "startup. Are you open to a quick chat this week to discuss?"},
    {"id": "m9", "from": "billing@cloudhost.com", "date": "2026-06-25", "label": None,
     "subject": "Your invoice #4471 is ready",
     "body": "Your monthly invoice of €42.00 is now available. No action needed; "
             "payment will be charged automatically on 2026-07-01."},
    {"id": "m10", "from": "mark@acme.com", "date": "2026-06-24", "label": None,
     "subject": "Re: onboarding checklist for new hire",
     "body": "Quick one — can you confirm whether the new hire starts Monday or Tuesday? "
             "I need to book the laptop and access by tomorrow."},
    {"id": "m11", "from": "deals@megastore.shop", "date": "2026-06-23", "label": None,
     "subject": "🔥 48h FLASH SALE — 70% off everything!!!",
     "body": "Don't miss our biggest sale ever! Click now to claim 70% off. "
             "Limited stock, offer ends tonight!!!"},
    {"id": "m12", "from": "team@github.com", "date": "2026-06-23", "label": None,
     "subject": "[acme/api] PR #218 was merged",
     "body": "Your pull request 'Add retry logic to client' was merged into main by david. "
             "No action required."},
]

_SEED_EVENTS = [
    {"id": "e1", "summary": "Daily standup", "start": "2026-06-29T09:00", "end": "2026-06-29T09:15"},
    {"id": "e2", "summary": "1:1 with manager", "start": "2026-06-29T14:00", "end": "2026-06-29T14:30"},
    {"id": "e3", "summary": "Lunch with Sara", "start": "2026-06-30T12:00", "end": "2026-06-30T13:00"},
    {"id": "e4", "summary": "Dentist", "start": "2026-07-01T16:00", "end": "2026-07-01T17:00"},
]

INBOX: list[dict] = []
EVENTS: list[dict] = []
DRAFTS: list[dict] = []
SENT: list[dict] = []


def reset_world() -> None:
    """Restore inbox / calendar to seed data and clear drafts + sent."""
    INBOX[:] = copy.deepcopy(_SEED_INBOX)
    EVENTS[:] = copy.deepcopy(_SEED_EVENTS)
    DRAFTS[:] = []
    SENT[:] = []


reset_world()


# ── Email tools (read / label are ungated) ────────────────────
@tool
def list_messages(days: int = 7) -> dict:
    """List inbox messages: id, sender, date, subject, and current triage label."""
    return {"messages": [{"id": m["id"], "from": m["from"], "date": m["date"],
                          "subject": m["subject"], "label": m["label"]} for m in INBOX]}


@tool
def get_thread(message_id: str) -> dict:
    """Read the full body of one inbox message by its id."""
    for m in INBOX:
        if m["id"] == message_id:
            return m
    return {"error": f"no message {message_id}"}


@tool
def label_message(message_id: str, label: Literal["NEEDS_REPLY", "FYI", "SPAM"]) -> dict:
    """Apply a triage label to a message."""
    for m in INBOX:
        if m["id"] == message_id:
            m["label"] = label
            return {"id": message_id, "label": label}
    return {"error": f"no message {message_id}"}


# ── Calendar tools (read is ungated) ──────────────────────────
@tool
def list_events(days_ahead: int = 14) -> dict:
    """List upcoming calendar events."""
    return {"events": EVENTS}


@tool
def suggest_time(attendees: list[str], duration_minutes: int) -> dict:
    """Suggest the next free slot that fits all attendees."""
    return {"start": "2026-06-30T10:00", "end": "2026-06-30T11:00", "attendees": attendees}


# ── Gated tools — require human approval before they fire ─────
# Creating a draft is itself gated: the human reviews (and can edit) the full
# draft before it is saved. Nothing is ever sent.
@tool
def create_draft(to: str, subject: str, body: str) -> dict:
    """GATED: prepare an email reply for review. Saved as a draft after approval; never sent."""
    draft = {"id": f"d{len(DRAFTS) + 1}", "to": to, "subject": subject,
             "body": body, "sent": False}
    DRAFTS.append(draft)
    return {"draft_id": draft["id"], "to": to, "subject": subject}


@tool
def create_event(summary: str, start: str, end: str, attendees: list[str]) -> dict:
    """GATED: create a calendar event. Requires human approval."""
    event = {"id": f"e{len(EVENTS) + 1}", "summary": summary,
             "start": start, "end": end, "attendees": attendees}
    EVENTS.append(event)
    return {"created": True, **event}


GATED_TOOLS = {"create_draft": create_draft, "create_event": create_event}

EMAIL_TOOLS = [list_messages, get_thread, label_message]   # ungated (run by ToolNode)
CALENDAR_TOOLS = [list_events, suggest_time]               # ungated (run by ToolNode)


# ── Accessors for the UI ──────────────────────────────────────
def current_inbox() -> list[dict]:
    return INBOX


def current_events() -> list[dict]:
    return EVENTS


def current_drafts() -> list[dict]:
    return DRAFTS


# ─────────────────────────────────────────────────────────────
# LLM — one shared instance; sub-agents get tool-bound copies.
# (Gated tools are bound so the agent can *propose* them; the
#  HITL gate intercepts the proposal before it executes.)
# ─────────────────────────────────────────────────────────────
_llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=0,
    base_url=BASE_URL,
    api_key=os.environ.get("OPENAI_API_KEY", "missing-key"),
)
email_llm = _llm.bind_tools(EMAIL_TOOLS + [create_draft])    # create_draft is gated
calendar_llm = _llm.bind_tools(CALENDAR_TOOLS + [create_event])


# ─────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────
_SUPERVISOR_PROMPT = """\
You are the triage supervisor for a personal assistant.
Classify the user's request into exactly one intent:
  "email"         - only needs email operations (read / draft / triage)
  "calendar"      - only needs calendar operations (availability / events)
  "cross_service" - needs BOTH (e.g. find a slot AND send a confirmation email)
Reply with ONLY JSON: {"intent": "<email|calendar|cross_service>"}
"""


def supervisor_node(
    state: AgentState,
) -> Command[Literal["email_agent", "calendar_agent", "cross_service"]]:
    """Classify intent and route to the right specialist."""
    response = _llm.invoke([SystemMessage(content=_SUPERVISOR_PROMPT), *state["messages"]])
    try:
        intent = json.loads(response.content).get("intent", "email")
    except (json.JSONDecodeError, AttributeError):
        intent = "email"
    route = {"email": "email_agent", "calendar": "calendar_agent", "cross_service": "cross_service"}
    goto = route.get(intent, "email_agent")
    print(f"[supervisor] intent={intent!r} -> {goto}")
    return Command(update={"intent": intent}, goto=goto)


_EMAIL_PROMPT = """\
You are the Email Agent. Tools: list_messages, get_thread, label_message,
create_draft.

Follow this workflow with tool calls (do not just describe it in text):
1. Call list_messages.
2. For EVERY message, call label_message with NEEDS_REPLY, FYI, or SPAM.
   You must actually call the tool for each one before writing any summary.
3. When the user asks you to reply to a specific message, call get_thread to
   read its full body, then call create_draft.
4. Write a COMPLETE, ready-to-send reply in create_draft: a greeting using the
   sender's name, 2-4 sentences that directly address their message, and a
   sign-off ("Best regards,\\nMy Assistant"). create_draft is reviewed by a
   human and saved as a draft — nothing is sent.
Use only email addresses that appear in the inbox. After the tool calls are
done, give a short one-line-per-message summary.
"""


def email_agent_node(state: AgentState) -> dict:
    """Email specialist: triage, draft, propose send."""
    print("[email_agent] thinking...")
    return {"messages": [email_llm.invoke([SystemMessage(content=_EMAIL_PROMPT), *state["messages"]])]}


_CALENDAR_PROMPT = """\
You are the Calendar Agent. Tools: list_events, suggest_time, create_event.
Check the schedule, suggest a free slot (weekdays 09:00-17:00), then call
create_event to propose it (a human approves first). Always include summary,
start, end, and attendees.
"""


def calendar_agent_node(state: AgentState) -> dict:
    """Calendar specialist: check availability, propose an event."""
    print("[calendar_agent] thinking...")
    return {"messages": [calendar_llm.invoke([SystemMessage(content=_CALENDAR_PROMPT), *state["messages"]])]}


def cross_service_node(state: AgentState) -> Command[Literal["calendar_agent"]]:
    """Route cross-service tasks: calendar first (find a slot), then email follows."""
    print("[cross_service] calendar first, then email")
    hint = HumanMessage(content=(
        "[hint] Cross-service task: first find a calendar slot, then the email "
        "agent will draft a confirmation. Start with calendar."
    ))
    return Command(update={"messages": [hint]}, goto="calendar_agent")


def _route_agent(state: AgentState, tool_node: str) -> str:
    """Shared routing: gated call -> HITL, normal tool -> tools, else END."""
    calls = getattr(state["messages"][-1], "tool_calls", []) or []
    if not calls:
        return END
    if any(c["name"] in GATED_TOOLS for c in calls):
        return "hitl_gate"
    return tool_node


def email_should_continue(state: AgentState) -> Literal["email_tools", "hitl_gate", "__end__"]:
    return _route_agent(state, "email_tools")


def calendar_should_continue(state: AgentState) -> Literal["calendar_tools", "hitl_gate", "__end__"]:
    return _route_agent(state, "calendar_tools")


def hitl_gate_node(state: AgentState) -> dict:
    """Pause and ask a human to approve the proposed gated action."""
    calls = getattr(state["messages"][-1], "tool_calls", []) or []
    pending = next(
        ({"tool": c["name"], "args": c["args"]} for c in calls if c["name"] in GATED_TOOLS),
        None,
    )
    print(f"[hitl_gate] pending action: {pending}")
    # interrupt() suspends the graph until resumed with Command(resume=...).
    # resume may be "approve"/"reject", or a dict {"decision", "args"} so the
    # human can edit the draft/event before approving.
    decision = interrupt({"prompt": "Approve this action? (approve/reject)", "pending_action": pending})
    if isinstance(decision, dict):
        if decision.get("args"):
            pending = {"tool": pending["tool"], "args": decision["args"]}
        decision = decision.get("decision", "reject")
    return {"hitl_decision": decision, "pending_action": pending}


def after_hitl(state: AgentState) -> Literal["execute_action", "__end__"]:
    if state.get("hitl_decision") == "approve":
        print("[hitl] approved")
        return "execute_action"
    print("[hitl] rejected")
    return END


def execute_action_node(state: AgentState) -> dict:
    """Fire the gated tool now that a human has approved it."""
    pending = state["pending_action"]
    result = GATED_TOOLS[pending["tool"]].invoke(pending["args"])
    print(f"[execute] {pending['tool']} -> {result}")
    summary = ", ".join(f"{k}={v}" for k, v in pending["args"].items())
    return {"messages": [AIMessage(content=f"✅ {pending['tool']} completed ({summary}).")]}


# ─────────────────────────────────────────────────────────────
# Graph assembly
# ─────────────────────────────────────────────────────────────
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("supervisor", supervisor_node)
    g.add_node("email_agent", email_agent_node)
    g.add_node("calendar_agent", calendar_agent_node)
    g.add_node("cross_service", cross_service_node)
    g.add_node("email_tools", ToolNode(EMAIL_TOOLS))
    g.add_node("calendar_tools", ToolNode(CALENDAR_TOOLS))
    g.add_node("hitl_gate", hitl_gate_node)
    g.add_node("execute_action", execute_action_node)

    g.add_edge(START, "supervisor")  # supervisor + cross_service route via Command()
    g.add_conditional_edges("email_agent", email_should_continue)
    g.add_edge("email_tools", "email_agent")
    g.add_conditional_edges("calendar_agent", calendar_should_continue)
    g.add_edge("calendar_tools", "calendar_agent")
    g.add_conditional_edges("hitl_gate", after_hitl)
    g.add_edge("execute_action", END)

    # Checkpointing lets the graph pause at the HITL gate and resume later.
    return g.compile(checkpointer=MemorySaver())


graph = build_graph()


# ─────────────────────────────────────────────────────────────
# Demo runner
# ─────────────────────────────────────────────────────────────
def run_task(task: str, thread_id: str | None = None) -> str:
    """Run one task. Returns the thread_id (reuse it to resume after HITL)."""
    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    print(f"\n{'='*60}\nTASK: {task}\n{'='*60}")
    for chunk in graph.stream(
        {"messages": [HumanMessage(content=task)], "intent": None,
         "pending_action": None, "hitl_decision": None},
        config=config, stream_mode="updates",
    ):
        _print_chunk(chunk)
    return thread_id


def resume(thread_id: str, decision: Literal["approve", "reject"]) -> None:
    """Resume a graph paused at the HITL gate with a human decision."""
    config = {"configurable": {"thread_id": thread_id}}
    print(f"\n[resume] decision={decision!r}")
    for chunk in graph.stream(Command(resume=decision), config=config, stream_mode="updates"):
        _print_chunk(chunk)


def _print_chunk(chunk: dict) -> None:
    for key, update in chunk.items():
        if key == "__interrupt__":  # graph paused at the HITL gate
            print("  -> ⏸  paused for human approval (resume with approve/reject)")
            continue
        if not isinstance(update, dict):
            continue
        for msg in update.get("messages", []):
            calls = getattr(msg, "tool_calls", None)
            if calls:
                print(f"  -> tool calls: {[c['name'] for c in calls]}")
            elif getattr(msg, "content", ""):
                print(f"  -> {str(msg.content)[:120]}")


def save_graph_html(path: str = "graph.live.html") -> None:
    """Write a self-contained HTML page that renders the live graph as a
    Mermaid diagram, straight from the compiled StateGraph (opens in any
    browser — nothing is uploaded). The curated diagram lives in
    architecture.html / architecture.md."""
    mermaid = graph.get_graph().draw_mermaid()
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Personal Assistant — graph</title>
<style>body{{font-family:system-ui;margin:2rem;background:#0f172a;color:#e2e8f0}}
h1{{font-weight:600}} .mermaid{{background:#fff;border-radius:12px;padding:1.5rem}}</style>
<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<script>mermaid.initialize({{startOnLoad:true,theme:'default'}});</script>
</head><body>
<h1>Personal Assistant Agent — LangGraph</h1>
<pre class="mermaid">
{mermaid}</pre>
</body></html>"""
    with open(path, "w") as f:
        f.write(html)
    print(f"Wrote {path} — open it in a browser to view the graph.")


if __name__ == "__main__":
    if "--graph" in sys.argv:
        save_graph_html()
        sys.exit(0)

    # 1) Pure email — triage the inbox.
    run_task("Sort my inbox: which need a reply, which are FYI, which are spam?")

    # 2) Cross-service — find a slot, then a draft + send proposal pauses at HITL.
    tid = run_task("Anna wants a 1-hour project sync this week. Find a slot and draft a reply.")
    resume(tid, "approve")  # try "reject" to see the gate stop the action
