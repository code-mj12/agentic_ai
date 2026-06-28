# Personal Assistant Agent

A research project on **Personal Assistant Agents** — their architectures,
privacy/safety properties, and evaluation — with a small, runnable LangGraph
demo that grounds the survey in working code.

The research is the main deliverable; the implementation is a compact
illustration of the survey's core ideas, plus a model-comparison case study.

## Repository layout

```
.
├── src/                      # the implementation
│   ├── assistant.py          # LangGraph agent (supervisor → agents → HITL gate)
│   ├── app.py                # Gradio web demo
│   └── compare_models.py     # offline model benchmark
├── data/                     # test suites (real + synthetic) — see data/README.md
├── docs/
│   ├── architecture.md       # diagram + node/state/tool reference
│   ├── architecture.html     # standalone styled graph diagram
│   └── report.md             # evaluation case study (criteria + results vs research)
├── research/                 # survey PDF + seminar slides (main deliverable)
├── benchmark_results.json    # stored benchmark, shown in the app's 📊 Models tab
├── requirements.txt
└── .env.example
```

## The demo

An email/calendar assistant that shows three ideas from the survey:

- **Architecture** — a *supervisor* classifies each request and routes it to a
  specialist *email* or *calendar* agent (or both, for cross-service tasks).
  Each agent runs a ReAct tool loop.
- **Privacy / safety** — write actions (`create_draft`, `create_event`) pass
  through a **human-in-the-loop (HITL) gate**: you review and can **edit** the
  draft/event before it's saved. The agent never sends email (read + draft only).
- **Statefulness** — the graph checkpoints, so it can pause at the gate and
  resume on a human decision.

It uses a rich in-memory inbox/calendar, so no email/calendar accounts or Docker
are needed — only access to the LLM endpoint. The LLM runs on the University of
Passau OpenAI-compatible endpoint (`llms.innkube.fim.uni-passau.de`, default
model `qwen3-next-80b-a3b-instruct`). See [docs/report.md](docs/report.md) for the
model-comparison case study and [docs/architecture.md](docs/architecture.md) for
the graph diagram.

## Setup

```bash
pip3 install -r requirements.txt
cp .env.example .env        # then put your access key in .env (do this once)
```

Run all commands **from the repository root**.

## Run the visual demo (recommended)

```bash
python3 src/app.py
```

Open the printed URL (http://127.0.0.1:7860). Type a request — or click an
example — and watch:

- the supervisor route it and the agents call tools step by step in the
  **live trace** panel;
- the **Inbox / Calendar / Drafts** tables update as the agent acts — emails get
  triage labels, drafts get written, approved events appear on the calendar.
  **Click an inbox row to read the full email.**
- a **review panel** appear at the human-in-the-loop gate: for a draft you get
  editable To / Subject / Body fields — edit, then **Approve** to save it (or
  **Reject**). Nothing is sent.

Use **🔄 Refresh** to redraw the tables and **♻️ Reset demo data** to restore
the starting inbox/calendar.

## Compare models

The app's **📊 Models** tab shows a stored benchmark across **five capability
dimensions** (the survey's taxonomy): state-reading (spam detection on a recent
real [phishing dataset](https://huggingface.co/datasets/darkknight25/phishing_benign_email_dataset)),
planning, tool-use, drafting, and safety (prompt-injection detection) — with
per-capability scores, latency, token cost, bar charts, and analysis. Test data
lives in [data/](data/). To (re)generate the results:

```bash
python3 src/compare_models.py     # writes benchmark_results.json
```

The app reads `benchmark_results.json` at startup, so the tab loads instantly —
no live model calls. Full case study in [docs/report.md](docs/report.md).

## Run in the terminal / view the graph

```bash
python3 src/assistant.py          # runs two demo tasks in the console
python3 src/assistant.py --graph  # writes graph.live.html (diagram of the live graph)
```
