# Pathos AI
### Advanced Stateful Medical RAG Assistant

Pathos AI is a portfolio-grade reference implementation of a clinical-education
RAG assistant, built to demonstrate production engineering practices around
LLM orchestration, retrieval quality, data privacy, and safety guardrails —
not a certified medical device, and not a substitute for professional care.

```
FastAPI · LangGraph · LangChain · Qdrant (hybrid search) · FlashRank
SQLAlchemy (async) · JWT auth · OpenTelemetry / LangSmith · Tailwind-styled SPA
```

---

## Table of contents
1. [System design](#system-design)
2. [Repository structure](#repository-structure)
3. [Trade-offs & architecture decisions](#trade-offs--architecture-decisions)
4. [Local installation](#local-installation)
5. [Free-tier setup (no paid API key required)](#free-tier-setup-no-paid-api-key-required)
6. [Running the test suite](#running-the-test-suite)
7. [API overview](#api-overview)
8. [Safety & privacy posture](#safety--privacy-posture)
9. [Known limitations](#known-limitations)
10. [Troubleshooting (Windows-specific)](#troubleshooting-windows-specific)

---

## System design

Pathos AI's core is a **LangGraph state machine**, not a linear chain. Every
message flows through a fixed sequence of nodes, each independently traced,
independently testable, and connected by explicit (including conditional)
edges:

```
UserInput → PIIMasking → InputGuardrail ──blocked──────────────┐
                              │allowed                          │
                              ▼                                 │
                        QueryRewrite                            │
                              │                                 │
                              ▼                                 │
                       HybridRetrieval                          │
                              │                                 │
                              ▼                                 │
                    CrossEncoderRerank                          │
                              │                                 │
                              ▼                                 │
                        Generation ◀──regenerate (≤1)──┐         │
                              │                        │         │
                              ▼                        │         │
                       OutputGuardrail ─────────────────┘         │
                              │allowed                            │
                              ▼                                   │
                        PIIUnmasking ◀────────────────────────────┘
                              │
                              ▼
                       StreamedOutput
```

The full Mermaid state diagram, a request-lifecycle sequence diagram, and
the PII trust-boundary diagram all live in **[`docs/architecture.md`](docs/architecture.md)** —
render it directly on GitHub or in any Mermaid-compatible viewer.

**Why a graph matters here, concretely:** a message flagged by the input
guardrail (e.g. a prompt-injection attempt, or crisis language) must never
touch the vector store or the LLM at all — it short-circuits straight to a
safe response. A generation that fails the output guardrail (a definitive
diagnosis, an exact dosage) gets exactly one bounded retry with a stricter
system prompt, then falls back to a safe templated answer rather than ever
streaming a rejected draft. Modeling this as `if/else` inside one giant
function is how that logic silently rots; modeling it as a graph makes the
safety-critical control flow the thing you can point at in a diagram, write
a unit test against, and trace in production.

---

## Repository structure

```
pathos-ai/
├── app/
│   ├── main.py                   # FastAPI app, middleware, exception handlers
│   ├── config.py                 # Pydantic v2 settings & env validation
│   ├── schemas.py                # All API/internal Pydantic contracts
│   ├── database.py               # Async SQLAlchemy engine/session
│   ├── routers/                  # HTTP surface — thin, no business logic
│   │   ├── auth.py
│   │   ├── chat.py
│   │   └── reports.py
│   ├── services/                 # Business logic / orchestration
│   │   ├── auth_service.py
│   │   ├── privacy_engine.py     # PII masking/unmasking
│   │   ├── guardrails.py         # Clinical safety rules
│   │   └── retrieval_service.py  # Hybrid search + RRF fusion
│   ├── engines/                  # Heavier ML/AI components
│   │   ├── llm_graph.py          # The LangGraph workflow itself
│   │   ├── embeddings.py         # Dense embedder + BM25 sparse encoder
│   │   └── reranker.py           # FlashRank cross-encoder
│   ├── models/
│   │   └── db_models.py          # SQLAlchemy ORM models
│   └── core/
│       ├── deps.py               # FastAPI auth dependency
│       ├── logging_config.py     # Structured, PII-safe JSON logging
│       └── telemetry.py          # OpenTelemetry + LangSmith hooks
├── frontend/                     # Tailwind-styled SPA (vanilla JS, no build step)
│   ├── index.html
│   └── static/{css,js}/
├── tests/
│   ├── conftest.py
│   └── test_llm.py                # Guardrail + privacy engine unit tests
├── docs/
│   └── architecture.md            # Mermaid diagrams + component table
├── requirements.txt
├── docker-compose.yml             # Qdrant + Postgres + API, one command
├── Dockerfile
└── .env.example
```

**Layer separation, and why it's drawn this way:** routers only parse/validate
HTTP input and call a service — they contain zero business logic, so the same
`retrieval_service.hybrid_search()` call is reachable from a router, a
background job, or a test, identically. Services own business rules
(guardrails, privacy). Engines own the heavier AI/ML machinery (embeddings,
reranking, the graph itself) — this line exists because engines are where
you'd swap a whole implementation (Qdrant → Pinecone, FlashRank → a hosted
reranker) without a service or router ever needing to change.

---

## Trade-offs & architecture decisions

### FastAPI over Flask
Pathos AI's request lifecycle is dominated by I/O waits — the LLM call, the
Qdrant round-trip, the reranker, the database. Flask's default WSGI model
handles concurrent requests by adding worker processes/threads, which is
fine for CPU-bound work but wasteful here: a single slow LLM call blocks a
whole worker for its entire duration. FastAPI's native `async`/`await`
support means a single worker can hold dozens of in-flight chat requests
concurrently, each one yielding the event loop back during its network
waits. That's also why the chat endpoint is a native `StreamingResponse`
over SSE rather than a polling endpoint — the transport matches the
async-first execution model instead of fighting it.

### Qdrant/Pinecone over raw Chroma
Chroma is excellent for prototyping — embed, upsert, done. It falls short
for Pathos AI's specific requirement: **hybrid search**. Dense-only search
misses exact-match cases that matter clinically (a drug name, an ICD-style
code, an exact symptom term) where lexical overlap is the real signal, not
semantic similarity. Qdrant has first-class native sparse vector support
(so dense + sparse can live in one collection, one round-trip) and a
clearer path to production (payload filtering, horizontal scaling,
snapshotting) than Chroma's embedded-first design offers. Pinecone is kept
as a configurable alternate backend (`VECTOR_BACKEND=pinecone`) for teams
already standardized on it — the `RetrievalService` interface is written so
swapping backends never touches `llm_graph.py`.

### LangGraph over a simple sequential chain
A chain (`retriever | reranker | prompt | llm`) assumes every request takes
the same path. Clinical dialogue does not — see the "why a graph matters"
paragraph above. Concretely, a plain chain has no clean way to express "this
request should never reach the LLM at all" or "this LLM output needs exactly
one bounded retry with a different prompt" without ad-hoc control flow
wrapped around the chain from the outside. LangGraph makes both branches
first-class, typed edges in the graph definition itself, which is also what
makes the graph's node-by-node execution directly traceable (see the
Pipeline Trace panel in the UI) — the graph structure *is* the observability
structure.

### Reversible, in-request PII tokenization over full anonymization
Pathos AI never sends raw PII to an external LLM provider or writes it to
logs/database — but it also never permanently destroys it, because the
authenticated user still needs to see their own name/phone number/etc.
reflected back in the UI. The `pii_map` produced by `PrivacyEngine.mask()`
lives only in-process, for the duration of a single request, and is
explicitly never persisted (see the trust-boundary diagram in
`docs/architecture.md`). This is a deliberate middle ground between "send
everything" (a real compliance risk) and "destroy everything" (a bad user
experience, and it makes debugging a specific user's issue much harder).

### Cross-encoder reranking as a hard requirement, not an optimization
Hybrid retrieval alone tends to return "plausible" chunks — lexically or
semantically close, but not necessarily the *most* responsive to the actual
question. Skipping reranking and just concatenating the top-N hybrid hits
is "context stuffing": it dilutes the generation prompt with tangential
material and burns tokens that could go toward more relevant context. The
reranker here also enforces a hard token budget (`max_context_tokens`) —
it will return fewer than `rerank_top_k` chunks rather than blow the budget,
which is a deliberate quality-over-quantity choice for a domain where a
diluted, unfocused answer is a worse failure mode than a shorter one.

### Async SQLAlchemy + `create_all` over Alembic (for this reference build)
The project ships with `Base.metadata.create_all()` at startup rather than
Alembic migrations, to keep local setup to "clone, `pip install`, run" with
zero extra tooling. A real production deployment should replace this with
Alembic-managed migrations before the first schema change ships — the
models themselves are already standard SQLAlchemy 2.0 declarative classes,
so wiring in Alembic later is additive, not a rewrite.

---

## Local installation

> **Prerequisite:** Python 3.11+, Docker (for Qdrant/Postgres), and an
> OpenAI or Anthropic API key.

```bash
# 1. Clone and enter the project
git clone <your-fork-url> pathos-ai && cd pathos-ai

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# then edit .env: set JWT_SECRET_KEY, OPENAI_API_KEY (or ANTHROPIC_API_KEY)

# 5. Start Qdrant + Postgres (or skip Postgres and keep the default SQLite)
docker compose up -d qdrant postgres

# 6. Run the API (auto-creates tables on first boot)
uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000** — register an account, and start chatting.
API docs (Swagger UI) are available at **http://localhost:8000/api/docs**
in any non-production environment.

### Loading a knowledge base
`RetrievalService.index_documents()` is the entry point for populating
Qdrant + the BM25 index. A minimal loader script (chunk your source
documents, then call `await retrieval_service.index_documents(chunks)`) is
the fastest way to get a working knowledge base indexed — this is left as
an integration point rather than a fixed script, since real deployments
each have their own clinical content source and chunking strategy.

---

## Free-tier setup (no paid API key required)

You don't need a paid OpenAI account to run Pathos AI. Two swaps make the
entire stack free:

**1. Chat generation → Groq.** Groq gives out a free API key (no credit
card) and speaks the exact same API format as OpenAI, so no code changes
are needed — just point the existing OpenAI client at a different URL.

**2. Embeddings → local, via `fastembed`.** Instead of calling an embedding
API at all, this runs a small ONNX model directly on your machine. Zero
cost, zero network calls after the first model download, and — importantly
for Windows users — no PyTorch dependency (see the troubleshooting section
below for why that matters).

### Setup

1. Get a free key at **https://console.groq.com/keys** (`gsk_...`).
2. Install the local embedding dependency:
   ```bash
   pip install fastembed==0.3.6
   ```
3. In `.env`, set:
   ```
   LLM_PROVIDER=openai
   OPENAI_API_KEY=gsk_your-groq-key-here
   OPENAI_BASE_URL=https://api.groq.com/openai/v1
   GENERATION_MODEL=llama-3.3-70b-versatile

   EMBEDDING_PROVIDER=local
   LOCAL_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
   LOCAL_EMBEDDING_DIMENSIONS=384
   ```
   Yes, `LLM_PROVIDER` stays `openai` even though the key is from Groq —
   that field controls which *client library/request format* is used
   (OpenAI's), not literally which company issued the key.
4. Restart the server so the new `.env` values are actually loaded (see the
   troubleshooting note on `--reload` below — this step trips people up).

### Limitations of the free tier

- Groq's free tier is rate-limited (roughly ~1,000 requests/day depending
  on the model) — fine for development and demos, not for a public
  production deployment.
- If you exceed Groq's limit mid-session, Google AI Studio's Gemini free
  tier is another genuinely free, no-card option, though it needs a
  slightly different `OPENAI_BASE_URL` (Gemini's OpenAI-compatibility
  layer) rather than reusing the Groq config as-is.

---

```bash
pytest tests/ -v
```

`tests/test_llm.py` validates, with zero dependency on a live LLM or
database:
- Crisis/self-harm language is detected and blocked with crisis resources
- Prompt-injection and jailbreak attempts are blocked
- Exact-dosage requests are flagged (not blocked) so generation stays
  conservative
- Draft answers containing a definitive diagnosis or an exact dosage
  trigger a `REGENERATE` verdict
- Hedged, appropriately-disclaimed clinical language passes cleanly
- PII masking/unmasking round-trips correctly, and never leaks raw PII
  into the masked text

---

## API overview

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/auth/register` | Create an account |
| `POST` | `/api/v1/auth/login` | Obtain an access/refresh token pair |
| `POST` | `/api/v1/auth/refresh` | Exchange a refresh token for a new pair |
| `POST` | `/api/v1/chat/stream` | Send a message; returns an SSE token stream |
| `GET` | `/api/v1/chat/sessions` | List the authenticated user's chat sessions |
| `POST` | `/api/v1/reports/preview` | Structured JSON preview of a session report |
| `POST` | `/api/v1/reports/download` | Download the session report as a PDF |
| `GET` | `/api/health` | Liveness check |

---

## Safety & privacy posture

- **PII never crosses the LLM boundary.** Every message is masked before
  retrieval/generation and before any log line is emitted (see
  `app/services/privacy_engine.py` and `app/core/logging_config.py`).
- **No definitive diagnoses, ever.** The output guardrail regenerates (once)
  or falls back to a safe template rather than let a diagnostic-sounding
  sentence reach the user.
- **No exact dosages, ever.** Same mechanism as above.
- **Crisis language is detected before retrieval or generation runs**, and
  routes straight to a resource-first response.
- **Every guardrail decision is audit-logged** (`GuardrailAuditLog`) —
  independent of the (already-masked) message content, so a compliance
  review can verify the safety layer actually fired without needing access
  to conversation content at all.

This system is a **portfolio/reference implementation**. It is not
FDA-cleared, HIPAA-certified, or clinically validated, and should not be
used to make real health decisions.

---

## Known limitations

- Person-name PII detection is heuristic (pattern-based), not a full NER
  model — see the note at the top of `privacy_engine.py` for the intended
  swap-in point (spaCy / a hosted PII service).
- The in-memory BM25 mirror in `RetrievalService` is meant for small/medium
  corpora; very large corpora should move sparse scoring into Qdrant's
  native sparse vector index instead.
- Query rewriting for multi-turn follow-ups is a cheap heuristic concat,
  not an LLM-based rewrite — swap-in point is `query_rewrite_node` in
  `app/engines/llm_graph.py`.
- Database migrations use `create_all` rather than Alembic — see the
  trade-offs section above before taking this to a real production
  environment with evolving schema.

---

## Troubleshooting (Windows-specific)

These are real issues encountered running Pathos AI on Windows/PowerShell,
kept here so they don't need to be rediscovered.

### `.venv/bin/activate` not recognized
That's the Linux/Mac path. On Windows PowerShell, use:
```powershell
.venv\Scripts\Activate.ps1
```
If PowerShell blocks script execution, run this once per session:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### `pip install -r requirements.txt` fails with a `ResolutionImpossible` error
Usually a version pin that's too strict for a transitive dependency (e.g.
`langchain-openai` needing a newer `openai` than the pin allows). Loosen
the pin to a range, e.g. `openai>=1.52.0,<2.0.0`, rather than an exact `==`.

### `ValueError: password cannot be longer than 72 bytes` during registration
This is **not** about your actual password — it's `passlib==1.7.4` running
a broken internal self-test against `bcrypt>=4.1.0`. Fix by pinning:
```
bcrypt==4.0.1
```
This can resurface any time you rebuild your virtualenv from scratch if
`requirements.txt` doesn't have that pin — check it's actually there with
`Select-String -Path requirements.txt -Pattern "bcrypt"` if it comes back.

### `OSError: [WinError 206] The filename or extension is too long`
Hit when installing `sentence-transformers` (pulls in PyTorch) or `onnx`
on Windows — both ship deeply nested internal folders that exceed
Windows' default 260-character path limit, especially painful with
Microsoft Store Python's already-long install path
(`...PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\...`).
Permanent fix — **enable long paths system-wide** (admin PowerShell,
then reboot, not just relaunch the terminal):
```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```
Pathos AI also sidesteps this for local embeddings specifically by using
`fastembed` (ONNX, no PyTorch) instead of `sentence-transformers`.

### `docker: command not recognized` / Docker Compose can't reach the daemon
Docker Desktop isn't installed, or isn't running. Install it, open it, and
wait for the whale icon in the system tray to settle before running
`docker compose up -d qdrant` again. `dockerDesktopLinuxEngine: the system
cannot find the file specified` specifically means the engine isn't
started yet — the CLI is installed but the background service isn't.

### `pydantic_settings.SettingsError: error parsing value for field "allowed_cors_origins"`
Pydantic-settings tries to JSON-decode env values for any `list[str]`-typed
field, which breaks on a plain comma-separated string. Fixed in this
codebase by storing it as a raw string field (`allowed_cors_origins_raw`)
with a `@property` that splits it — no per-field annotation workarounds
needed.

### Edited `.env` but nothing changed after saving
**`uvicorn --reload` only watches Python source files — it does not watch
`.env`.** Editing `.env` while the server is running has no effect until
you fully stop it (`Ctrl+C`, confirm the prompt returns) and start it again.
This is the single most common "I changed the config but it's still doing
the old thing" cause.

### 401 Unauthorized on every request, right after restarting the server
The browser has a stale access token saved in `localStorage` from a
previous session (e.g. before you recreated the database or changed
`JWT_SECRET_KEY`). Clear it via the browser DevTools console (`F12` →
Console tab):
```javascript
localStorage.removeItem('pathos_access_token')
```
then refresh — you should land back on the login screen.

### `AuthenticationError: Incorrect API key provided` mentioning `platform.openai.com`, even though you're using Groq
If the error message points to OpenAI's own site, the request went to
`api.openai.com` directly instead of your configured `OPENAI_BASE_URL` —
almost always because the server process was started before `.env` was
saved (see the `--reload` note above). Do a full restart.

### `.venv\Scripts\Activate.ps1` : "The module '.venv' could not be loaded"
PowerShell can't find anything at that path and misinterprets it as a
module name. Run `dir` first to confirm you're actually in the project
folder and that a `.venv` directory exists there — if it doesn't, create
it fresh with `python -m venv .venv` from that exact location, and never
move/copy that folder afterward (the activation scripts inside hard-code
an absolute path at creation time).

### Code edits don't seem to take effect at all
If `.env` changes are correctly loading (confirmed via a full restart) but
behavior still doesn't match the code you think is on disk, check for a
second, older copy of the project folder — e.g. from re-extracting a zip
into a folder that already had one, producing a nested
`pathos-ai\pathos-ai\` structure where your editor and terminal are
pointed at different copies. Confirm with:
```powershell
Get-ChildItem -Path C:\Users\<you>\Downloads -Recurse -Directory -Filter "pathos-ai" | Select-Object FullName
```