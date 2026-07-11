// ============================================================================
// Pathos AI — Frontend Application Logic
// ============================================================================

const API_BASE = "/api/v1";

let state = {
  accessToken: localStorage.getItem("pathos_access_token") || null,
  currentSessionId: null,
  authMode: "login",
};

const PIPELINE_NODES = [
  { key: "pii_masking", label: "PII Masking" },
  { key: "input_guardrail", label: "Input Guardrail" },
  { key: "query_rewrite", label: "Query Rewrite" },
  { key: "hybrid_retrieval", label: "Hybrid Retrieval" },
  { key: "cross_encoder_rerank", label: "Cross-Encoder Rerank" },
  { key: "generation", label: "Generation" },
  { key: "output_guardrail", label: "Output Guardrail" },
  { key: "pii_unmasking", label: "PII Unmasking" },
];

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------

function toggleAuthMode() {
  state.authMode = state.authMode === "login" ? "register" : "login";
  document.getElementById("login-form").style.display = state.authMode === "login" ? "block" : "none";
  document.getElementById("register-form").style.display = state.authMode === "register" ? "block" : "none";
  document.getElementById("auth-error").textContent = "";
}

async function handleRegister() {
  const full_name = document.getElementById("register-name").value.trim();
  const email = document.getElementById("register-email").value.trim();
  const password = document.getElementById("register-password").value;
  const errorEl = document.getElementById("auth-error");
  errorEl.textContent = "";

  try {
    const res = await fetch(`${API_BASE}/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ full_name, email, password }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Registration failed.");
    }
    toggleAuthMode();
    errorEl.style.color = "var(--pathos-success)";
    errorEl.textContent = "Account created — please sign in.";
  } catch (e) {
    errorEl.textContent = e.message;
  }
}

async function handleLogin() {
  const email = document.getElementById("login-email").value.trim();
  const password = document.getElementById("login-password").value;
  const errorEl = document.getElementById("auth-error");
  errorEl.textContent = "";

  try {
    const res = await fetch(`${API_BASE}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Login failed.");
    }
    const data = await res.json();
    state.accessToken = data.access_token;
    localStorage.setItem("pathos_access_token", data.access_token);
    showApp();
  } catch (e) {
    errorEl.textContent = e.message;
  }
}

function showApp() {
  document.getElementById("auth-screen").style.display = "none";
  document.getElementById("app-shell").style.display = "grid";
  loadSessions();
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

async function loadSessions() {
  try {
    const res = await authedFetch(`${API_BASE}/chat/sessions`);
    if (!res.ok) return;
    const sessions = await res.json();
    const listEl = document.getElementById("session-list");
    listEl.innerHTML = "";
    sessions.forEach((s) => {
      const item = document.createElement("div");
      item.className = "session-item" + (s.id === state.currentSessionId ? " active" : "");
      item.textContent = s.title;
      item.onclick = () => selectSession(s.id, s.title);
      listEl.appendChild(item);
    });
  } catch (e) {
    console.error("Failed to load sessions", e);
  }
}

function selectSession(id, title) {
  state.currentSessionId = id;
  document.getElementById("session-title").textContent = title;
  document.getElementById("messages").innerHTML = "";
  loadSessions();
}

function startNewSession() {
  state.currentSessionId = null;
  document.getElementById("session-title").textContent = "New conversation";
  document.getElementById("messages").innerHTML = "";
  resetTracePanel();
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

function handleTextareaKeydown(evt) {
  if (evt.key === "Enter" && !evt.shiftKey) {
    evt.preventDefault();
    sendMessage();
  }
}

function appendMessage(role, content, extra) {
  const messagesEl = document.getElementById("messages");
  const row = document.createElement("div");
  row.className = `msg-row ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "msg-avatar";
  avatar.textContent = role === "user" ? "You" : "P";

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";

  if (extra && extra.riskLevel && extra.riskLevel !== "none" && extra.riskLevel !== "RiskLevel.NONE") {
    const badge = document.createElement("div");
    const level = String(extra.riskLevel).replace("RiskLevel.", "").toLowerCase();
    badge.className = `risk-badge ${level}`;
    badge.textContent = `${level} risk`;
    bubble.appendChild(badge);
  }

  const textEl = document.createElement("div");
  textEl.className = "msg-text";
  textEl.textContent = content;
  bubble.appendChild(textEl);

  if (extra && extra.citations && extra.citations.length > 0) {
    const toggle = document.createElement("div");
    toggle.className = "sources-toggle";
    toggle.innerHTML = `📚 Sources cited (${extra.citations.length}) <span>▾</span>`;

    const panel = document.createElement("div");
    panel.className = "sources-panel";
    extra.citations.forEach((c) => {
      const chip = document.createElement("div");
      chip.className = "source-chip";
      chip.innerHTML = `<strong>${escapeHtml(c.document_title)}</strong> — ${escapeHtml(c.text.slice(0, 140))}...`;
      panel.appendChild(chip);
    });

    toggle.onclick = () => panel.classList.toggle("open");
    bubble.appendChild(toggle);
    bubble.appendChild(panel);
  }

  row.appendChild(avatar);
  row.appendChild(bubble);
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return textEl;
}

function appendLoadingSkeleton() {
  const messagesEl = document.getElementById("messages");
  const row = document.createElement("div");
  row.className = "msg-row assistant";
  row.id = "loading-skeleton";

  const avatar = document.createElement("div");
  avatar.className = "msg-avatar";
  avatar.textContent = "P";

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  bubble.style.minWidth = "180px";
  for (let i = 0; i < 3; i++) {
    const skRow = document.createElement("div");
    skRow.className = "skeleton-row";
    const bar = document.createElement("div");
    bar.className = "skeleton-bar";
    bar.style.width = i === 2 ? "60%" : "100%";
    skRow.appendChild(bar);
    bubble.appendChild(skRow);
  }

  row.appendChild(avatar);
  row.appendChild(bubble);
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function removeLoadingSkeleton() {
  const el = document.getElementById("loading-skeleton");
  if (el) el.remove();
}

async function sendMessage() {
  const textarea = document.getElementById("chat-textarea");
  const message = textarea.value.trim();
  if (!message) return;

  const sendBtn = document.getElementById("send-btn");
  sendBtn.disabled = true;
  textarea.value = "";

  appendMessage("user", message);
  resetTracePanel();
  appendLoadingSkeleton();

  try {
    const res = await authedFetch(`${API_BASE}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.currentSessionId, message }),
    });

    if (!res.ok || !res.body) {
      throw new Error("Pathos AI could not process this message.");
    }

    removeLoadingSkeleton();
    const textEl = appendMessage("assistant", "", { riskLevel: null, citations: [] });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let fullText = "";
    let meta = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const events = buffer.split("\n\n");
      buffer = events.pop();

      for (const evt of events) {
        const line = evt.replace(/^data: /, "").trim();
        if (!line) continue;
        const payload = JSON.parse(line);

        if (payload.type === "meta") {
          meta = payload;
          state.currentSessionId = payload.session_id;
          renderTrace(payload.trace);
          document.getElementById("stat-latency").textContent = `${payload.latency_ms} ms`;
          document.getElementById("stat-risk").textContent = String(payload.risk_level).replace("RiskLevel.", "");
          document.getElementById("trace-stats").style.display = "grid";
        } else if (payload.type === "token") {
          fullText += payload.content;
          textEl.textContent = fullText;
          document.getElementById("messages").scrollTop = document.getElementById("messages").scrollHeight;
        } else if (payload.type === "done") {
          if (meta && meta.citations && meta.citations.length > 0) {
            attachCitations(textEl, meta.citations);
          }
        }
      }
    }

    loadSessions();
  } catch (e) {
    removeLoadingSkeleton();
    appendMessage("assistant", `Something went wrong: ${e.message}`);
  } finally {
    sendBtn.disabled = false;
  }
}

function attachCitations(textEl, citations) {
  const bubble = textEl.parentElement;
  const toggle = document.createElement("div");
  toggle.className = "sources-toggle";
  toggle.innerHTML = `📚 Sources cited (${citations.length}) <span>▾</span>`;

  const panel = document.createElement("div");
  panel.className = "sources-panel";
  citations.forEach((c) => {
    const chip = document.createElement("div");
    chip.className = "source-chip";
    chip.innerHTML = `<strong>${escapeHtml(c.document_title)}</strong> — ${escapeHtml((c.text || "").slice(0, 140))}...`;
    panel.appendChild(chip);
  });

  toggle.onclick = () => panel.classList.toggle("open");
  bubble.appendChild(toggle);
  bubble.appendChild(panel);
}

// ---------------------------------------------------------------------------
// Trace panel
// ---------------------------------------------------------------------------

function resetTracePanel() {
  const stepsEl = document.getElementById("trace-steps");
  stepsEl.innerHTML = "";
  PIPELINE_NODES.forEach((node) => {
    const step = document.createElement("div");
    step.className = "trace-step pending";
    step.id = `trace-${node.key}`;
    step.innerHTML = `
      <div class="trace-dot"></div>
      <div class="trace-body">
        <div class="trace-name">${node.label}</div>
        <div class="trace-meta">pending</div>
      </div>`;
    stepsEl.appendChild(step);
  });
  document.getElementById("trace-stats").style.display = "none";
}

function renderTrace(trace) {
  if (!trace) return;
  trace.forEach((step, i) => {
    const el = document.getElementById(`trace-${step.node_name}`);
    if (!el) return;
    el.className = `trace-step ${step.status === "blocked" ? "blocked" : "active"}`;
    const meta = el.querySelector(".trace-meta");
    meta.textContent = `${step.status} · ${step.duration_ms.toFixed(1)}ms`;
  });
}

// ---------------------------------------------------------------------------
// Report modal
// ---------------------------------------------------------------------------

async function openReportModal() {
  if (!state.currentSessionId) {
    alert("Start a conversation before generating a report.");
    return;
  }
  document.getElementById("report-modal-backdrop").classList.add("open");
  const bodyEl = document.getElementById("report-modal-body");
  bodyEl.innerHTML = "Loading preview…";

  try {
    const res = await authedFetch(`${API_BASE}/reports/preview`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.currentSessionId, include_full_transcript: false }),
    });
    const data = await res.json();
    bodyEl.innerHTML = "";
    data.sections.forEach((s) => {
      const sec = document.createElement("div");
      sec.className = "report-section";
      sec.innerHTML = `<h4>${escapeHtml(s.heading)}</h4><p>${escapeHtml(s.body)}</p>`;
      bodyEl.appendChild(sec);
    });
    const disclaimer = document.createElement("div");
    disclaimer.style.cssText = "font-size:11px;color:var(--pathos-ink-muted);margin-top:10px;";
    disclaimer.textContent = data.disclaimer;
    bodyEl.appendChild(disclaimer);
  } catch (e) {
    bodyEl.innerHTML = "Failed to load report preview.";
  }
}

function closeReportModal() {
  document.getElementById("report-modal-backdrop").classList.remove("open");
}

async function downloadReport() {
  const res = await authedFetch(`${API_BASE}/reports/download`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: state.currentSessionId, include_full_transcript: false }),
  });
  const blob = await res.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `pathos-ai-report-${state.currentSessionId}.pdf`;
  a.click();
  window.URL.revokeObjectURL(url);
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

async function authedFetch(url, options = {}) {
  const headers = options.headers || {};
  headers["Authorization"] = `Bearer ${state.accessToken}`;
  return fetch(url, { ...options, headers });
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

resetTracePanel();
if (state.accessToken) {
  showApp();
}
