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
