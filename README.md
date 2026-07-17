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

Open the printed URL (http://127.0.0.1:7860). The app has **two top-level tabs**:

**💬 Assistant demo** — type a request (or click an example) and watch:

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

**📊 Model benchmark** — a full-width view (chat hidden) with example test items,
a colour-coded **capability-score heatmap**, overall/latency bar charts, and our
models **beside published research baselines**. Loads instantly from
`benchmark_results.json` — see the "Compare models" section below.

## Compare models

The app's **📊 Models** tab shows a stored benchmark of **4 models** across **five
hard suites**: **tool-argument extraction** (distractors), drafting, **prompt-injection
resistance** (2025 attacks, per-email), and two real intent benchmarks —
[Banking77](https://huggingface.co/datasets/mteb/banking77) (77 classes) and
[CLINC150](https://huggingface.co/datasets/clinc/clinc_oos) (150 + out-of-scope). Scores
are **graded (nothing at 0% or 100%)**: injection (per-email, 3 attack tiers) is the
discriminator, and the two largest models rank last. The results table shows our
per-model scores **beside published models from research papers** (fine-tuned RoBERTa /
BERT, GPT-4 zero-shot on Banking77 / CLINC150), and [docs/report.md](docs/report.md)
explains *technically why* the models score as they do vs. the literature. Test data in
[data/](data/). To (re)generate:

```bash
python3 src/compare_models.py     # writes benchmark_results.json
```

The app reads `benchmark_results.json` at startup, so the tab loads instantly —
no live model calls. Full case study in [docs/report.md](docs/report.md).

### Capabilities we tested and dropped

We prototyped nine capability suites and kept the **five** that are *hard, real, and
assistant-relevant*. The four we removed, and why:

| Dropped suite | Why removed |
|---|---|
| **Planning / intent routing** (email/calendar/cross-service) | A trivial 3-class version of intent classification — the real **Banking77** (77 classes) tests the same capability far harder; it was redundant. |
| **State-reading / BEC** (spear-phishing vs legit) | Synthetic and saturated (100% for 3 of 4 models); it didn't separate models. |
| **Tool selection** (pick the right function) | Every model scored 100% — too easy to be informative. |
| **Reasoning / MMLU-Pro** (expert MCQ) | General expert-knowledge QA, not a personal-assistant capability — out of scope for an email/calendar agent. |

The kept five map to what the assistant actually does: **read & classify** (Banking77,
CLINC150), **act precisely** (tool-argument extraction), **write** (drafting), and stay
**safe** (injection resistance).

## Run in the terminal / view the graph

```bash
python3 src/assistant.py          # runs two demo tasks in the console
python3 src/assistant.py --graph  # writes graph.live.html (diagram of the live graph)
```
