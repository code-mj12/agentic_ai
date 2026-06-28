# Test datasets

Stored separately from the benchmark code (`src/compare_models.py`) so the data and
the harness are independent.

## `phishing_email_sample.jsonl` — recent, open source

A small **balanced sample (8 SPAM/phishing + 8 LEGIT/benign)** drawn from the
**2025**
[`darkknight25/phishing_benign_email_dataset`](https://huggingface.co/datasets/darkknight25/phishing_benign_email_dataset)
on Hugging Face. Each line is one email:

```json
{"id": "...", "subject": "...", "body": "...", "label": "SPAM" | "LEGIT"}
```

We deliberately use a **recent** dataset (2025) rather than the classic 2006
Enron-Spam corpus: Enron is almost certainly in the models' pre-training data, so
a benchmark on it risks measuring *memorisation* rather than ability. The
phishing/benign pairs here are also intentionally subtle (e.g. a phishing
"Invoice #3921 Overdue" vs a legitimate "Your GitHub Invoice"), which tests the
model rather than surface keywords. Bodies are whitespace-normalised and
truncated to ~240 chars to keep prompts small and the suite fast/testable.

Regenerate a fresh balanced sample:

```bash
curl -s "https://datasets-server.huggingface.co/rows?dataset=darkknight25/phishing_benign_email_dataset&config=default&split=train&offset=0&length=100" -o /tmp/phish.json
# then build 8 phishing + 8 benign, truncate bodies, write this file
```

## Synthetic suites (hand-authored, small)

No clean public dataset exists for these assistant-specific capabilities, so they
are hand-authored, clearly marked synthetic, and kept small/diagnostic. They
cover the survey's capability taxonomy beyond spam detection:

| File | Dimension | Items | Label / gold |
|---|---|---|---|
| `routing_suite.json` | Planning | 8 | intent: email / calendar / cross_service |
| `toolcall_suite.json` | Tool use | 8 | correct tool name for the request |
| `draft_suite.json` | Generation | 4 | sender name (drafts scored by structural completeness) |
| `injection_suite.json` | Safety / hand-off | 8 | INJECTION / CLEAN (prompt-injection detection) |

See `../docs/report.md` for the criteria catalogue and how each is scored.
