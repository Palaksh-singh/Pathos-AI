# Pathos AI — System Architecture

## 1. LangGraph Clinical Routing State Machine

Every inbound message is compiled into a `PathosGraphState` object and passed through
a directed graph of nodes. Each node is a pure function `(state) -> state` (or an async
node for I/O-bound work), which makes the pipeline replayable, testable in isolation,
and fully traceable in LangSmith / Arize Phoenix.

```mermaid
stateDiagram-v2
    [*] --> UserInput

    UserInput --> PIIMasking: raw message received

    PIIMasking --> InputGuardrail: PII spans replaced\nwith stable tokens

    InputGuardrail --> Blocked: unsafe / malicious intent detected
    InputGuardrail --> QueryRewrite: passes safety screen

    QueryRewrite --> HybridRetrieval: standalone query\n(conversation-aware)

    HybridRetrieval --> CrossEncoderRerank: dense (embeddings) +\nsparse (BM25) candidates

    CrossEncoderRerank --> ContextAssembly: top-k reranked chunks\n(context-stuffing guard: token budget cap)

    ContextAssembly --> Generation: grounded prompt +\nchat history + disclaimers

    Generation --> OutputGuardrail: streamed LLM tokens

    OutputGuardrail --> Regenerate: diagnostic claim /\ndosage / unsafe pattern found
    Regenerate --> Generation: max 1 retry with\nstricter system prompt

    OutputGuardrail --> PIIUnmasking: passes safety screen
    PIIUnmasking --> ResponseAssembly: restore user-facing\nplaceholders (not raw PII)

    ResponseAssembly --> StreamedOutput: attach citations,\ndisclaimer, trace metadata

    Blocked --> StreamedOutput: canned safe-refusal\n+ crisis/escalation resources

    StreamedOutput --> [*]

    note right of PIIMasking
        Reversible tokenization:
        PERSON_1, PHONE_1, MRN_1 ...
        Raw PII never leaves this
        process boundary.
    end note

    note right of OutputGuardrail
        Regex + LLM-judge pass:
        - No definitive diagnosis
        - No exact drug dosages
        - Must carry disclaimer
    end note
```

## 2. Why a graph, not a chain

A linear chain assumes the happy path always applies. Clinical dialogue does not:
a message can be unsafe before retrieval ever runs, a generation can fail guardrails
and need exactly one bounded retry, and a blocked message must short-circuit straight
to output without ever touching the vector store or the LLM. LangGraph models these as
explicit edges and conditional branches instead of `if/else` sprawl inside a single
function, which is what makes the state machine above directly executable as code
(see `app/engines/llm_graph.py`) instead of just documentation.

## 3. Request lifecycle (sequence view)

```mermaid
sequenceDiagram
    participant U as User (Browser)
    participant API as FastAPI /chat/stream
    participant PE as Privacy Engine
    participant G as LangGraph Runtime
    participant VDB as Qdrant (Hybrid Index)
    participant RR as Cross-Encoder Reranker
    participant LLM as LLM Provider

    U->>API: POST /chat/stream {message, session_id}
    API->>API: JWT auth + rate limit
    API->>PE: mask_pii(message)
    PE-->>API: masked_message, pii_map
    API->>G: ainvoke(state)
    G->>G: input_guardrail_node
    alt blocked
        G-->>API: refusal + resources
    else allowed
        G->>VDB: hybrid_search(query)
        VDB-->>G: dense + sparse candidates
        G->>RR: rerank(candidates)
        RR-->>G: top_k chunks
        G->>LLM: stream(generation_prompt)
        LLM-->>G: token stream
        G->>G: output_guardrail_node (per-chunk + final)
    end
    G-->>API: final state (answer, citations, trace)
    API->>PE: unmask_pii(answer, pii_map)
    API-->>U: SSE stream (tokens, then sources + trace)
```

## 4. Component responsibilities

| Layer | Module | Responsibility |
|---|---|---|
| API | `app/main.py`, `app/routers/*` | HTTP/SSE surface, auth, request validation |
| Service | `app/services/privacy_engine.py` | PII detection, tokenization, reversible masking |
| Service | `app/services/guardrails.py` | Input/output clinical safety checks |
| Service | `app/services/retrieval_service.py` | Hybrid search orchestration against Qdrant |
| Engine | `app/engines/llm_graph.py` | LangGraph state machine (the diagram above, as code) |
| Engine | `app/engines/embeddings.py` | Dense embedding + BM25 sparse encoding |
| Engine | `app/engines/reranker.py` | Cross-encoder reranking (FlashRank) |
| Data | `app/models/db_models.py`, `app/database.py` | SQLAlchemy ORM, session/user/message persistence |
| Core | `app/core/telemetry.py` | OpenTelemetry + LangSmith tracing hooks |
| Core | `app/core/logging_config.py` | Structured, PII-safe JSON logging |

## 5. Data privacy boundary

```mermaid
flowchart LR
    subgraph Trust Boundary: Pathos AI Backend
        A[Raw user message] --> B[Privacy Engine\nPII Masking]
        B --> C[Masked message\nPERSON_1, MRN_1, PHONE_1]
    end
    C --> D[LLM Provider\n(OpenAI / Anthropic API)]
    C --> E[Structured Logs / LangSmith Trace]
    B -.->|pii_map kept in-memory\nper-request only| F[(Never persisted\nto DB or logs)]
```

Raw PII is only ever held in-process, in memory, for the lifetime of a single request
(the `pii_map` dict). It is not written to the database, not written to logs, and not
sent to any external LLM provider — only the masked, tokenized surrogate text crosses
that boundary. Unmasking happens after generation, purely for rendering the response
back to the authenticated user who owns the session.
