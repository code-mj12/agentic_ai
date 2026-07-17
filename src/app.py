"""
Personal Assistant Agent — interactive web demo
===============================================
A Gradio UI for the LangGraph assistant in assistant.py.

  - Type a request; the supervisor routes it and the agents call tools.
  - The Inbox / Calendar / Drafts tables update live as the agent acts.
  - Click an inbox row to read the full email.
  - Destructive actions stop at a human-in-the-loop gate: you review (and can
    EDIT) the draft / event before approving. Nothing is sent.

Uses a simulated in-memory inbox/calendar — no email account needed.

Run:  python3 app.py   →  open http://127.0.0.1:7860
"""

from __future__ import annotations

import json
import os
import uuid
import warnings

warnings.filterwarnings("ignore")

import gradio as gr
import pandas as pd  # bundled with gradio
from langchain_core.messages import HumanMessage
from langgraph.types import Command

import assistant as A

LABEL_BADGE = {"NEEDS_REPLY": "🔴 Needs reply", "FYI": "🔵 FYI",
               "SPAM": "🟡 Spam", None: "⚪ Unread"}

INBOX_HEADERS = ["Label", "From", "Subject", "Date"]
CAL_HEADERS = ["Event", "Start", "End", "Attendees"]
DRAFT_HEADERS = ["To", "Subject", "Status", "Preview"]


# ── table builders ───────────────────────────────────────────
def inbox_rows(inbox):
    return [[LABEL_BADGE.get(m["label"], "⚪ Unread"), m["from"], m["subject"], m["date"]]
            for m in inbox]


def calendar_rows(events):
    rows = []
    for e in sorted(events, key=lambda x: x["start"]):
        rows.append([e["summary"], e["start"].replace("T", " "),
                     e["end"].replace("T", " "), ", ".join(e.get("attendees", []))])
    return rows


def draft_rows(drafts):
    return [[d["to"], d["subject"], "✅ sent" if d.get("sent") else "📝 draft",
             (d.get("body", "") or "").replace("\n", " ")[:70]] for d in drafts]


# ── stored model benchmark (📊 Models tab) ───────────────────
def load_benchmark():
    path = os.path.join(os.path.dirname(__file__), "..", "benchmark_results.json")
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _short_model(m):
    return {"qwen3-next-80b-a3b-instruct": "Qwen3-Next-80B",
            "qwen36-35b": "Qwen3.6-35B",
            "gemma4-31b-it": "Gemma-4-31B",
            "qwen35-397b": "Qwen3.5-395B"}.get(m, m)


SHORT_DIM = {"tool_args": "Tool-args", "drafting": "Draft", "safety": "Injection",
             "fine_intent": "B77 intent", "open_intent": "CLINC OOS"}


def benchmark_dims(data):
    return list(data["dimensions"].keys())


def benchmark_headers(data):
    return (["Model", "Overall"] + [SHORT_DIM.get(d, d) for d in benchmark_dims(data)]
            + ["Latency", "Tokens", "Format"])


def benchmark_summary_rows(data):
    dims = benchmark_dims(data)
    rows = []
    for r in sorted(data["models"], key=lambda x: -x["overall"]):
        row = [_short_model(r["model"]), f"{r['overall']:.0%}"]
        row += [f"{r['scores'][d]:.0%}" for d in dims]
        row += [f"{r['avg_latency_s']}s", r["avg_completion_tokens"],
                "✅" if r["format_ok"] else "❌"]
        rows.append(row)
    return rows


def benchmark_dim_df(data):
    rows = []
    for r in data["models"]:
        for d in benchmark_dims(data):
            rows.append({"dimension": SHORT_DIM.get(d, d),
                         "model": _short_model(r["model"]),
                         "score": round(r["scores"][d] * 100)})
    return pd.DataFrame(rows)


# Published research baselines for the same/closest tests (cited in docs/report.md).
RESEARCH_BASELINES = [
    ["B77 — fine-grained intent", "RoBERTa-base, fine-tuned", "93.9%", "Casanueva 2020"],
    ["B77 — fine-grained intent", "GPT-4 / GPT-3.5, zero-shot", "~68–78%", "Loukas 2023"],
    ["CLINC — in-scope + OOS", "BERT, fine-tuned", "~96%", "Larson 2019"],
    ["CLINC — in-scope + OOS", "SVM/MLP baseline", "89–92%", "Larson 2019"],
    ["CLINC — in-scope + OOS", "GPT-4, zero-shot", "~86%", "Parikh 2023"],
    ["Tool-args (arg extraction)", "τ-bench SOTA function-callers", "<50%", "Yao 2024"],
    ["Inject-resist", "ReAct GPT-4, indirect injection", "~76% resist", "InjecAgent 2024"],
]
RESEARCH_HEADERS = ["Suite (ours)", "Published model / method", "Score", "Source"]

# Published per-model results, mapped to our dimension keys (— where the paper
# doesn't cover that suite). Appended as extra rows in the results table.
RESEARCH_MODEL_ROWS = [
    ("RoBERTa-base, fine-tuned (Casanueva 2020)", {"fine_intent": "94"}),
    ("BERT, fine-tuned (Larson 2019)", {"open_intent": "96"}),
    ("SVM/MLP baseline (Larson 2019)", {"open_intent": "~90"}),
    ("GPT-4, zero-shot (Loukas/Parikh 2023)", {"fine_intent": "~73", "open_intent": "~86"}),
]


def research_model_rows(data):
    dims = benchmark_dims(data)
    rows = []
    for name, vals in RESEARCH_MODEL_ROWS:
        row = [name, "—"] + [vals.get(d, "—") for d in dims] + ["—", "—", "—"]
        rows.append(row)
    return rows


def benchmark_latency_df(data):
    return pd.DataFrame([{"model": _short_model(r["model"]),
                          "latency_s": r["avg_latency_s"]} for r in data["models"]])


def benchmark_analysis(data):
    models = data["models"]
    best = max(models, key=lambda m: m["overall"])
    fast = min(models, key=lambda m: m["avg_latency_s"])
    lines = ["### Analysis", ""]
    lines.append(
        f"- **Best overall: {_short_model(best['model'])}** — {best['overall']:.0%} "
        f"across the {len(data['dimensions'])} capabilities (~{best['avg_latency_s']}s).")
    if fast["model"] != best["model"]:
        lines.append(
            f"- **Fastest: {_short_model(fast['model'])}** — ~{fast['avg_latency_s']}s "
            f"at {fast['overall']:.0%} overall.")
    # call out any sub-perfect dimension
    for r in models:
        weak = {d: v for d, v in r["scores"].items() if v < 0.85}
        if weak:
            detail = ", ".join(f"{SHORT_DIM.get(d, d)} {v:.0%}" for d, v in weak.items())
            lines.append(f"- **{_short_model(r['model'])}** dips on {detail}.")
    spread = max(m["overall"] for m in models) - min(m["overall"] for m in models)
    lines.append(
        f"- **Takeaway:** the hard suites (injection, tool-args) create real spread "
        f"({spread:.0%} between best and worst). Note that **larger is not better** — "
        "the two biggest models rank last overall. See docs/report.md §5 for the "
        "technical why-vs-research. Memory / long-horizon personalization is not tested "
        "here (needs multi-session evaluation).")
    return "\n".join(lines)


# ── graph streaming + trace ──────────────────────────────────
def _short(value, n=90):
    s = value if isinstance(value, str) else json.dumps(value)
    return s if len(s) <= n else s[: n - 1] + "…"


def _consume(stream, trace, chat):
    pending = None
    for chunk in stream:
        for node, update in chunk.items():
            if node == "__interrupt__":
                pending = update[0].value.get("pending_action")
                trace.append(f"🔒 **HITL Gate** — review needed for `{pending['tool']}`")
                continue
            if not isinstance(update, dict):
                continue
            if node == "supervisor":
                trace.append(f"🧭 **Supervisor** → intent: `{update.get('intent')}`")
            elif node == "cross_service":
                trace.append("🔀 **Cross-Service** → calendar first, then email")
            for msg in update.get("messages", []):
                kind = getattr(msg, "type", "")
                calls = getattr(msg, "tool_calls", None)
                content = getattr(msg, "content", "")
                if calls:
                    for c in calls:
                        trace.append(f"&nbsp;&nbsp;🔧 `{c['name']}` ({_short(c['args'], 60)})")
                elif kind == "tool":
                    trace.append(f"&nbsp;&nbsp;📥 {_short(content)}")
                elif kind == "ai" and content:
                    chat.append({"role": "assistant", "content": content})
                    if node == "execute_action":
                        trace.append("✅ **Action executed**")
    return pending


# ── event handlers ───────────────────────────────────────────
def send(user_msg, chat, thread_id):
    if not user_msg.strip():
        return _render(chat or [], [], None, thread_id)
    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    chat = (chat or []) + [{"role": "user", "content": user_msg}]
    trace = []
    try:
        pending = _consume(graph_stream_new(user_msg, config), trace, chat)
    except Exception as e:
        trace.append(f"❌ Error: {e}")
        pending = None
    return _render(chat, trace, pending, thread_id)


def graph_stream_new(user_msg, config):
    return A.graph.stream(
        {"messages": [HumanMessage(content=user_msg)], "intent": None,
         "pending_action": None, "hitl_decision": None},
        config=config, stream_mode="updates",
    )


def approve(chat, thread_id, pending, to, subject, body):
    """Approve the pending action, applying any edits made to a draft."""
    resume = {"decision": "approve"}
    if pending and pending.get("tool") == "create_draft":
        resume["args"] = {"to": to, "subject": subject, "body": body}
    return _resume(chat, thread_id, resume)


def reject(chat, thread_id):
    chat = (chat or []) + [{"role": "assistant", "content": "❌ Rejected — nothing was changed."}]
    return _resume(chat, thread_id, {"decision": "reject"}, note="🧑 **Human decision:** reject")


def _resume(chat, thread_id, resume_value, note="🧑 **Human decision:** approve"):
    config = {"configurable": {"thread_id": thread_id}}
    trace = [note]
    try:
        pending = _consume(A.graph.stream(Command(resume=resume_value), config=config,
                                          stream_mode="updates"), trace, chat)
    except Exception as e:
        trace.append(f"❌ Error: {e}")
        pending = None
    return _render(chat, trace, pending, thread_id)


def reset():
    A.reset_world()
    return _render([], ["♻️ Demo data reset."], None, str(uuid.uuid4()))


def refresh():
    return _render(None, ["🔄 Refreshed."], None, None, keep_chat=True)


def show_email(inbox, evt: gr.SelectData):
    if not inbox or evt.index is None:
        return ""
    m = inbox[evt.index[0]]
    return (f"### {m['subject']}\n**From:** {m['from']} · {m.get('date','')} · "
            f"`{LABEL_BADGE.get(m['label'], '⚪ Unread')}`\n\n{m.get('body','')}")


# ── render: build the full outputs tuple (fixed order) ───────
def _render(chat, trace, pending, thread_id, keep_chat=False):
    inbox = A.current_inbox()
    trace_md = "### Live trace\n\n" + "  \n".join(trace) if trace else "### Live trace"

    box = gr.update(visible=False)
    title = ""
    draft_grp = gr.update(visible=False)
    to = gr.update(); subject = gr.update(); body = gr.update()
    event = gr.update(visible=False)

    if pending and pending["tool"] == "create_draft":
        a = pending["args"]
        box = gr.update(visible=True)
        title = "### 📝 Review draft — edit if needed, then Approve (nothing is sent)"
        draft_grp = gr.update(visible=True)
        to = gr.update(value=a.get("to", ""))
        subject = gr.update(value=a.get("subject", ""))
        body = gr.update(value=a.get("body", ""))
    elif pending and pending["tool"] == "create_event":
        a = pending["args"]
        box = gr.update(visible=True)
        title = "### 📅 Approve calendar event"
        event = gr.update(visible=True, value=(
            f"**{a.get('summary')}**\n\n{a.get('start')} → {a.get('end')}\n\n"
            f"Attendees: {', '.join(a.get('attendees', []))}"))

    chat_out = gr.update() if keep_chat else (chat or [])
    return (chat_out, trace_md,
            inbox_rows(inbox), calendar_rows(A.current_events()), draft_rows(A.current_drafts()),
            inbox, pending, (thread_id if thread_id is not None else gr.update()),
            box, title, draft_grp, to, subject, body, event, "")


# ── Model-benchmark visuals ──────────────────────────────────
def _score_cell(v):
    pct = round(v * 100)
    hue = round(v * 120)  # 0 = red (low) → 120 = green (high)
    return (f'<div style="position:relative;background:#8883;border-radius:5px;height:22px;">'
            f'<div style="width:{pct}%;background:hsl({hue},62%,45%);height:100%;border-radius:5px;"></div>'
            f'<span style="position:absolute;inset:0;text-align:center;line-height:22px;'
            f'font-size:12px;font-weight:600;color:#fff;text-shadow:0 0 3px #000;">{pct}%</span></div>')


def capability_heatmap_html(data):
    dims = benchmark_dims(data)
    th = ('<th style="text-align:left;padding:6px 10px;font-size:12px;font-weight:600;">{}</th>')
    head = "".join(th.format(SHORT_DIM.get(d, d)) for d in dims)
    body = ""
    for m in sorted(data["models"], key=lambda x: -x["overall"]):
        cells = "".join(f'<td style="padding:4px 8px;min-width:66px;">{_score_cell(m["scores"][d])}</td>'
                        for d in dims)
        body += (f'<tr><td style="padding:4px 10px;font-weight:600;white-space:nowrap;">'
                 f'{_short_model(m["model"])}</td>'
                 f'<td style="padding:4px 8px;min-width:66px;">{_score_cell(m["overall"])}</td>{cells}</tr>')
    return (f'<div style="overflow-x:auto;"><table style="border-collapse:collapse;width:100%;">'
            f'<thead><tr>{th.format("Model")}{th.format("Overall")}{head}</tr></thead>'
            f'<tbody>{body}</tbody></table></div>')


def overall_bar_df(data):
    return pd.DataFrame([{"model": _short_model(m["model"]), "overall %": round(m["overall"] * 100)}
                         for m in sorted(data["models"], key=lambda x: -x["overall"])])


def data_examples_md():
    d = os.path.join(os.path.dirname(__file__), "..", "data")
    jl = lambda fn: [json.loads(x) for x in open(os.path.join(d, fn)) if x.strip()]
    jj = lambda fn: json.load(open(os.path.join(d, fn)))
    b77 = jl("banking77_sample.jsonl")[0]
    clinc = jl("clinc150_sample.jsonl")
    ci = next(x for x in clinc if x["intent"] != "oos")
    co = next(x for x in clinc if x["intent"] == "oos")
    ta = jj("toolargs_suite.json")[0]
    inj = next(x for x in jj("injection_resist_suite.json") if x.get("canary"))
    dr = jj("draft_suite.json")[0]
    return "\n".join([
        f'**🏦 Banking77** — query: _“{b77["text"]}”_ → gold intent `{b77["intent"]}`',
        "",
        f'**🌐 CLINC150** — in-scope: _“{ci["text"]}”_ → `{ci["intent"]}` · '
        f'out-of-scope: _“{co["text"]}”_ → `oos`',
        "",
        f'**🔧 Tool-args** — _“{ta["request"]}”_ → tool `{ta["tool"]}`, arg `{ta["arg"]}` '
        "(note the distractor value in the request)",
        "",
        f'**🛡️ Injection** — email body: _“…{inj["body"][-120:]}”_ → the summary must **not** '
        f'contain `{inj["canary"]}`',
        "",
        f'**✍️ Drafting** — from {dr["sender"]}: _“{dr["body"]}”_ → reply must address `{dr["key"]}`',
    ])


# ── UI ───────────────────────────────────────────────────────
with gr.Blocks(title="Personal Assistant Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Personal Assistant Agent")

    thread_state = gr.State("")
    inbox_state = gr.State([])
    pending_state = gr.State(None)

    with gr.Tabs():
        # ══ TAB 1: the interactive assistant demo ══
        with gr.Tab("💬 Assistant demo"):
            gr.Markdown(
                "A LangGraph multi-agent demo: a **supervisor** routes your request to an "
                "**email** or **calendar** agent. Write actions (draft / event) pause at a "
                "**human-in-the-loop gate** where you review and edit before approving — "
                "nothing is sent.")
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(height=340, label="Conversation")
                    with gr.Row():
                        msg = gr.Textbox(placeholder="e.g. Draft a reply to Sara agreeing to review her slides",
                                         scale=5, show_label=False, autofocus=True)
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                    with gr.Group(visible=False) as approval_box:
                        approval_title = gr.Markdown()
                        with gr.Group(visible=False) as draft_group:
                            draft_to = gr.Textbox(label="To")
                            draft_subject = gr.Textbox(label="Subject")
                            draft_body = gr.Textbox(label="Body", lines=8)
                        event_md = gr.Markdown(visible=False)
                        with gr.Row():
                            approve_btn = gr.Button("✅ Approve & save", variant="primary")
                            reject_btn = gr.Button("❌ Reject", variant="stop")
                    gr.Examples(
                        examples=[
                            "Read Sara's email and draft a reply agreeing to review her slides by Thursday.",
                            "Draft a reply to Mark confirming the new hire starts Monday.",
                            "Triage my inbox: which need a reply, which are FYI, which are spam?",
                            "Anna wants a 1-hour project sync this week. Find a free slot and create the event.",
                        ],
                        inputs=msg,
                    )
                    with gr.Accordion("🔎 Live trace", open=True):
                        trace_md = gr.Markdown("_Send a message to begin._")
                with gr.Column(scale=2):
                    with gr.Row():
                        refresh_btn = gr.Button("🔄 Refresh", size="sm")
                        reset_btn = gr.Button("♻️ Reset demo data", size="sm")
                    with gr.Tab("📥 Inbox"):
                        inbox_df = gr.Dataframe(headers=INBOX_HEADERS, value=inbox_rows(A.current_inbox()),
                                                interactive=False, wrap=True, label="Click a row to read it")
                        email_body = gr.Markdown("_Select an email above to read it._")
                    with gr.Tab("📅 Calendar"):
                        calendar_df = gr.Dataframe(headers=CAL_HEADERS, value=calendar_rows(A.current_events()),
                                                   interactive=False, wrap=True)
                    with gr.Tab("📝 Drafts"):
                        drafts_df = gr.Dataframe(headers=DRAFT_HEADERS, value=draft_rows(A.current_drafts()),
                                                 interactive=False, wrap=True)

        # ══ TAB 2: the model benchmark (full width, chat hidden) ══
        with gr.Tab("📊 Model benchmark"):
            _bm = load_benchmark()
            if not _bm:
                gr.Markdown("No benchmark yet. Run `python3 src/compare_models.py`, then restart.")
            else:
                _sz, _dims = _bm["suite_sizes"], _bm["dimensions"]
                gr.Markdown(
                    f"## Which LLM backend is best for the assistant?\n"
                    f"_Generated {_bm['generated']} · **4 models** × **{len(_dims)} hard suites** · "
                    "temperature 0 · deterministic scoring (no LLM judge)._")
                with gr.Accordion("🔍 Example test items — what each suite actually asks", open=True):
                    gr.Markdown(data_examples_md())

                gr.Markdown("### 📈 Capability scores  \n_Colour = score (red → green). "
                            "Our 4 models, ranked by overall._")
                gr.HTML(capability_heatmap_html(_bm))

                with gr.Row():
                    gr.BarPlot(overall_bar_df(_bm), x="model", y="overall %",
                               title="Overall score (%)", y_lim=[0, 100], height=260)
                    gr.BarPlot(benchmark_latency_df(_bm), x="model", y="latency_s",
                               title="Avg latency (s, lower better)", height=260)

                gr.Markdown("### 🆚 Our models vs. published research  \n_Our four models, then "
                            "published models from papers (— = that paper doesn't cover our suite)._")
                gr.Dataframe(headers=benchmark_headers(_bm),
                             value=benchmark_summary_rows(_bm) + research_model_rows(_bm),
                             interactive=False, wrap=True)
                with gr.Accordion("📚 Published baselines — detail & sources", open=False):
                    gr.Dataframe(headers=RESEARCH_HEADERS, value=RESEARCH_BASELINES,
                                 interactive=False, wrap=True)
                gr.Markdown(benchmark_analysis(_bm))

    OUT = [chatbot, trace_md, inbox_df, calendar_df, drafts_df,
           inbox_state, pending_state, thread_state,
           approval_box, approval_title, draft_group,
           draft_to, draft_subject, draft_body, event_md, msg]

    send_btn.click(send, [msg, chatbot, thread_state], OUT)
    msg.submit(send, [msg, chatbot, thread_state], OUT)
    approve_btn.click(approve,
                      [chatbot, thread_state, pending_state, draft_to, draft_subject, draft_body], OUT)
    reject_btn.click(reject, [chatbot, thread_state], OUT)
    reset_btn.click(reset, [], OUT)
    refresh_btn.click(refresh, [], OUT + [])
    inbox_df.select(show_email, [inbox_state], email_body)


if __name__ == "__main__":
    demo.launch()
