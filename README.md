# finplat

AI Financial Crime Intelligence & Investigation Platform — **V0.1**.

This is the backend foundation only: FastAPI + PostgreSQL, JWT auth,
multi-tenant isolation, and a provider-agnostic AI gateway (Groq active,
Gemini/Cerebras present as inactive reference adapters). It is forked from
the same foundation in `production-ai-platform` (aiplat)'s Steps 1-2, with
no financial-crime domain logic yet — that comes in later V0.x steps.

## Stack

FastAPI, SQLAlchemy 2.x, Alembic, PostgreSQL, JWT auth (PyJWT), bcrypt
password hashing (passlib), pytest, `openai` SDK (pointed at Groq's
OpenAI-compatible API). `google-genai` (Gemini) is also installed, for the
inactive Gemini reference adapter.

## Project layout

```
app/
  models/          organizations, users, traces
  repositories/    Tenant-scoped query helpers (TenantScopedRepository)
  routers/         auth, users, ai
  schemas/         Pydantic request/response models
  gateway/         Provider-agnostic model gateway
    base.py          ModelProvider interface + GatewayResponse
    dependency.py     FastAPI dependency wiring the configured provider
    providers/groq.py      Groq adapter — ACTIVE
    providers/gemini.py    Gemini adapter — inactive reference
    providers/cerebras.py  Cerebras adapter — inactive reference
  config.py        Settings (env vars / .env)
  database.py      Engine/session setup
  dependencies.py  get_current_user / require_role auth dependencies
  security.py      Password hashing + JWT helpers
  main.py          FastAPI app
alembic/           Migrations (tracked as code)
tests/             pytest suite
```

## Setup

### 1. Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Environment variables

```bash
cp .env.example .env
```

Fill in `DATABASE_URL` and `JWT_SECRET`. `GROQ_API_KEY` is only needed for
`/ai/ask` — get a free key at https://console.groq.com/keys.

### 3. PostgreSQL

Point `DATABASE_URL` at a local Postgres instance with a `finplat`
database created. No Docker requirement for V0.1 — a native local
Postgres install works fine.

### 4. Run migrations

```bash
alembic upgrade head
```

### 5. Run the API

```bash
uvicorn app.main:app --reload --port 18000
```

Port 8000 is deliberately avoided: on WSL2, it falls inside Windows'
dynamic port-exclusion range (roughly 7901-8000), which can make binds to
it fail or hang unpredictably — the same issue aiplat hit and documented.
18000 is outside that range.

Visit http://localhost:18000/docs.

### 6. Run tests

```bash
pytest -v
```

Tests run against the same Postgres server, in their own `<db>_test`
database (created automatically), each test isolated in a rolled-back
transaction. The Groq/Gemini/Cerebras adapters are mocked at the SDK
boundary — the suite never calls a real provider and needs no API key.

## Evals

`evals/` holds golden-case eval harnesses covering RAG retrieval quality
and investigation-agent trajectory/decision quality — separate from
pytest because they need real models and cost real Groq quota (for the
agent eval) or a real local embedding model (for the RAG eval). Detection
accuracy against ground truth already has its own real eval, unrelated to
this stage: `scripts/verify_detection.py` (precision/recall per method
and per pattern).

```bash
python evals/run_rag_evals.py    # no Groq needed, real embedding model
python evals/run_agent_evals.py  # real Groq calls, costs quota
```

Each seeds an ephemeral organization (via the existing `seed_corpus.py` /
`seed_fraud_data.py`), runs its cases, prints a `PASS`/`FAIL` line per
case plus an `N/M passed` summary, and deletes everything it created
afterward. Scoring logic lives in `evals/scoring.py` as pure functions,
unit-tested in `tests/test_eval_scoring.py` — the only eval-related
pytest coverage, since the runners themselves must stay out of the
automated suite.

**Results as of 2026-09-19** (run three times over two days; both
findings below reproduced consistently and are documented here rather
than fixed in this pass):

- RAG retrieval: **9/10** — `structuring_query` fails: the actual
  `structuring_smurfing.md` typology doc ranks 4th against a top-3
  threshold, outranked by two `case_001_structuring.md` case-writeup
  chunks and a funnel-accounts chunk. Case write-ups apparently use more
  query-similar phrasing than the typology definition itself for this
  topic.
- Investigation agent: **4/5** (real Groq). The one failure
  (`layering_origin_observed_behavior`) surfaced a real, intermittent
  prompt gap: after a human **approves** an escalated action, the agent
  sometimes forgets to call `update_case`, leaving the case stuck at
  `in_review` even though its own final message claims the action was
  completed — the same class of "narrating a status it didn't actually
  set" bug V0.4 fixed for the clean-case path, not fully closed for the
  approved-escalation path. It fires inconsistently (2 of 3
  approved-escalation cases in this run closed correctly).
  A second, more concerning issue was observed on an earlier run (not
  reproduced on every run): the agent called `request_human_approval` on
  an account with a **provably clean** profile — `run_fraud_model`
  returned `flagged_by: [], findings: []` and the transaction history was
  an ordinary recurring-deposit pattern with no counterparties at all.
  The escalation trigger in `app/agent/graph.py`'s `SYSTEM_PROMPT`
  ("if you believe a specific action is warranted... call
  request_human_approval") doesn't require grounding in an actual
  finding before escalating, and this is the observed consequence.
  Neither issue is fixed here — see the V0.6 report. A related
  observability gap: intermediate agent reasoning isn't captured
  anywhere (only tool calls and final state), so the model's actual
  justification for the false escalation can't be quoted or verified,
  only inferred from the absence of any suspicious tool output. Worth
  closing eventually, not built in this pass.
