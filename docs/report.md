# Case Study: Comparing LLM Backends for a Personal Assistant Agent

**Companion to** *Personal Assistant Agents: A Research Survey on Architectures,
Privacy, and Evaluation* — this report turns the survey's "evaluation" axis into
a concrete experiment, using the demo in this repo (`src/app.py`, `src/assistant.py`) as
the harness and the survey's **capability taxonomy** as the evaluation frame.

---

## 1. Problem and data

**Problem.** A personal assistant does not do one thing — the survey lists five
required capabilities (synthesising Li et al. 2024 and the WorkBench taxonomy):
**state-reading, planning, tool-use, memory, and human hand-off**. Our assistant
reads and triages mail, routes requests, calls tools, drafts replies, and stops
at a human-in-the-loop gate. The same LangGraph agent runs on any
OpenAI-compatible model, so the question is: **which backend best supports the
whole job — not just one slice of it?**

An earlier version of this benchmark only scored spam classification. That is
too narrow: it ignores planning, tool-use, drafting, and safety. This version
evaluates **five capability dimensions**.

**Data.** Stored separately under `data/` (see `data/README.md`):

| # | Dimension (survey capability) | Suite | Size | Source |
|---|---|---|---|---|
| 1 | State-reading | spam/phishing detection | 16 | **Real, open source** — 2025 [`darkknight25/phishing_benign_email_dataset`](https://huggingface.co/datasets/darkknight25/phishing_benign_email_dataset) |
| 2 | Planning | intent routing | 8 | synthetic |
| 3 | Tool use | correct function selection | 8 | synthetic |
| 4 | Generation | reply drafting | 4 | synthetic |
| 5 | Human hand-off / safety | prompt-injection detection | 8 | synthetic |

The state-reading suite uses a **recent (2025)** real dataset on purpose: the
classic Enron-Spam corpus (2006) is almost certainly in the models' pre-training
data, so scoring on it would measure *memorisation*, not ability. The other four
suites are small, hand-authored, and gold-labelled so scoring is deterministic
(no LLM judge — which the survey notes can itself be hijacked by prompt
injection, AgentDojo). The **memory** capability is deliberately *not* tested:
it needs multi-session evaluation, which the survey itself lists under "what
current benchmarks miss."

## 2. Tools under test

The "tools" compared are the **chat models** on the University of Passau endpoint
(`llms.innkube.fim.uni-passau.de`), each used unchanged as the assistant backend:

| Model | Params (active) | Context | License |
|---|---|---|---|
| `qwen3-next-80b-a3b-instruct` | 80B (A3B MoE) | 262k | Apache-2.0 |
| `qwen36-35b` | 35B (A3B MoE) | 65k | Apache-2.0 |
| `gemma4-31b-it` | 31B (dense) | 131k | Apache-2.0 |

Qwen3.6 is a *thinking* model that previously looped and overran the output
budget on long prompts; we disable thinking via
`chat_template_kwargs.enable_thinking=false`, after which it returns directly.
Out of scope: Qwen3.5-395B (excluded by request); the reranker/embedding models
(not chat).

## 3. Evaluation criteria catalogue

| # | Criterion | Definition | Why it matters | Measurement |
|---|---|---|---|---|
| C1 | **State-reading** | spam vs legit on real mail | safety-critical triage | accuracy, 16 real emails |
| C2 | **Planning** | request → right service route | wrong route → wrong agent | accuracy, 8 requests |
| C3 | **Tool use** | request → correct function | core of function-calling agents | accuracy, 8 requests |
| C4 | **Generation** | complete, personalised reply | the "draft a reply" job | structural completeness (greeting, sign-off, length, names) over 4 emails |
| C5 | **Safety / hand-off** | spot prompt-injection in mail | injected mail can hijack agents | accuracy, 8 cases (4 attacks / 4 clean) |
| C6 | **Latency** | wall-clock per request | >2–3 s hurts UX | mean over all suites + repeats |
| C7 | **Token cost** | mean completion tokens | proxy for € / throughput | from API usage |
| C8 | **Format reliability** | parseable JSON within budget | unparseable = unusable | all suites parsed within 2048 tokens |

C1–C5 are *capability quality* (mapping to the survey's taxonomy), C6–C8 are
*operational fitness*.

## 4. Method

Each model runs all five suites at **temperature 0**, each as one batched call
returning JSON, scored deterministically against gold (C4 uses structural
heuristics, not an LLM judge). Latency is averaged over the suite calls plus
repeats of the heaviest suite. Data is loaded from `data/`, separate from the
harness. Fully reproducible:

```bash
python3 src/compare_models.py    # loads data/, writes benchmark_results.json
```

Results are stored and surfaced live in the app's **📊 Models** tab.

## 5. Results

_Generated 2026-06-28 · temperature 0 · scores are % correct (C4 = completeness)._

| Model | Overall | C1 State | C2 Plan | C3 Tools | C4 Draft | C5 Safety | Latency | Tokens |
|---|---|---|---|---|---|---|---|---|
| **Qwen3.6-35B** | **100 %** | 100 % | 100 % | 100 % | 100 % | 100 % | **0.74 s** | 136 |
| **Gemma-4-31B** | **100 %** | 100 % | 100 % | 100 % | 100 % | 100 % | 2.87 s | 139 |
| **Qwen3-Next-80B** | 96 % | 100 % | 100 % | 100 % | 81 % | 100 % | 1.63 s | 119 |

### Comparison to research

Multi-task agent benchmarks in the survey report **much lower** absolute scores —
WorkBench (workplace email/calendar/CRM): GPT-4 **43 %**, Llama-2-70B 3 %;
τ-bench: SOTA function-callers **< 50 %**; on prompt injection, ReAct GPT-4 is
**24 % vulnerable** (InjecAgent) and best agents are attacked **< 25 %** before
defenses (AgentDojo). Our near-100 % scores are **not** a contradiction: those
benchmarks use long-horizon, multi-step, adversarial tasks, whereas our suites
are small, single-turn, and curated to be diagnostic. The value here is the
**relative** comparison of interchangeable backends on the assistant's own jobs,
plus the operational metrics — exactly the trade-off the survey says matters.

## 6. Analysis

- **All three clear the capability bar** on these suites — modern instruct models
  handle triage, routing, tool selection, drafting, and injection-spotting well.
  So **quality no longer separates them**, and the decision moves to C6–C8.
- **Qwen3.6-35B is the best pick here:** perfect on all five dimensions and the
  **fastest** (0.74 s) — but only after disabling its "thinking" mode, which
  otherwise made it loop and overrun the token budget. A concrete lesson: a
  capable model can be unusable until configured correctly (C8).
- **Gemma-4-31B** matches it on quality but is ~4× slower (2.9 s).
- **Qwen3-Next-80B** is fast and cheapest in tokens, but its drafts scored 81 %
  on the structural check (a reply that missed a greeting/sign-off element) — the
  only quality gap in the table.
- **Takeaway:** evaluating across the survey's capability taxonomy — not a single
  task — gives a defensible backend choice, and shows that **operational fitness
  (latency, cost, format reliability, correct configuration)** is the real
  differentiator once a model is capable enough.

## 7. Threats to validity

- **Small, partly synthetic suites:** four of five suites are hand-authored and
  small, so scores saturate near 100 %; they are diagnostic, not a leaderboard.
  Larger/harder data would re-introduce spread.
- **Drafting is scored structurally,** not semantically — it checks greeting,
  sign-off, length, and name, not whether the reply is *good*. A real evaluation
  would add human or (cautiously) LLM-judge ratings.
- **Safety is detection, not resistance:** C5 checks whether a model *recognises*
  injection, not whether it *refuses to act* on it inside the live agent loop.
- **Memory untested:** the survey's fifth capability needs multi-session
  evaluation, out of scope here.
- **Single endpoint, single day:** latency reflects this shared endpoint on
  2026-06-28.

## 8. Conclusion

Treating interchangeable LLMs as competing "tools" and scoring them against a
**capability-aligned criteria catalogue** — state-reading, planning, tool-use,
generation, and safety, plus latency/cost/format — gives a clear, defensible
backend choice and demonstrates the survey's central evaluation argument:
personal-assistant quality must be judged across **capabilities, cost, and
operational fitness together**, not on a single narrow task.

---

### Sources
- [darkknight25/phishing_benign_email_dataset (2025) — Hugging Face](https://huggingface.co/datasets/darkknight25/phishing_benign_email_dataset) — the real test data
- WorkBench (Styles et al., arXiv:2405.00823) — workplace email/calendar agent tasks
- τ-bench (Yao et al., arXiv:2406.12045) — tool-agent task success
- AgentDojo (Debenedetti et al., NeurIPS 2024) & InjecAgent (Zhan et al., ACL Findings 2024) — prompt-injection robustness
- [Trustworthiness Calibration for Phishing Detection with LLMs — arXiv 2511.04728](https://arxiv.org/pdf/2511.04728)
- [LLM Latency Benchmark by Use Case — AIMultiple](https://research.aimultiple.com/llm-latency-benchmark/)
