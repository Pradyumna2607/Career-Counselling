const API_BASE = "/api";

let currentUser = null;
let currentSessionId = null;

const chatScroll = document.getElementById("chatScroll");
const emptyState = document.getElementById("emptyState");
const sessionList = document.getElementById("sessionList");
const messageInput = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const suggestionChips = document.getElementById("suggestionChips");

// ------------------------------------------------------------------ //
// Bootstrap: verify auth, load sessions
// ------------------------------------------------------------------ //
(async function init() {
  const res = await fetch(`${API_BASE}/auth/me`, { credentials: "include" });
  const data = await res.json();
  if (!data.user) {
    window.location.href = "/index.html";
    return;
  }
  currentUser = data.user;
  document.getElementById("userName").textContent = currentUser.full_name;
  document.getElementById("userAvatar").textContent = currentUser.full_name.charAt(0).toUpperCase();

  await loadSessions();
})();

// ------------------------------------------------------------------ //
// Sessions
// ------------------------------------------------------------------ //
async function loadSessions() {
  const res = await fetch(`${API_BASE}/chat/sessions`, { credentials: "include" });
  const data = await res.json();
  renderSessionList(data.sessions || []);
}

function renderSessionList(sessions) {
  sessionList.innerHTML = "";
  sessions.forEach(s => {
    const item = document.createElement("div");
    item.className = "session-item" + (s.id === currentSessionId ? " active" : "");
    item.innerHTML = `<span>${escapeHtml(s.title || "New Chat")}</span><span class="session-del" data-id="${s.id}">✕</span>`;
    item.addEventListener("click", (e) => {
      if (e.target.classList.contains("session-del")) return;
      openSession(s.id);
    });
    item.querySelector(".session-del").addEventListener("click", async (e) => {
      e.stopPropagation();
      await fetch(`${API_BASE}/chat/sessions/${s.id}`, { method: "DELETE", credentials: "include" });
      if (currentSessionId === s.id) {
        currentSessionId = null;
        chatScroll.innerHTML = "";
        chatScroll.appendChild(emptyState);
      }
      loadSessions();
    });
    sessionList.appendChild(item);
  });
}

async function openSession(sessionId) {
  currentSessionId = sessionId;
  await loadSessions();
  const res = await fetch(`${API_BASE}/chat/sessions/${sessionId}/messages`, { credentials: "include" });
  const data = await res.json();
  chatScroll.innerHTML = "";
  (data.messages || []).forEach(renderMessage);
  scrollToBottom();
}

document.getElementById("newChatBtn").addEventListener("click", () => {
  currentSessionId = null;
  chatScroll.innerHTML = "";
  chatScroll.appendChild(emptyState);
  renderSessionList([...sessionList.children].map(() => null)); // noop visual reset
  loadSessions();
});

// ------------------------------------------------------------------ //
// Messaging
// ------------------------------------------------------------------ //
async function sendMessage(text) {
  if (!text.trim()) return;
  if (emptyState.parentElement) emptyState.remove();

  renderMessage({ sender: "user", text, timestamp: new Date().toISOString() });
  scrollToBottom();
  messageInput.value = "";
  autoGrow();

  const typingEl = renderTyping();
  scrollToBottom();
  sendBtn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/chat/message`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ text, session_id: currentSessionId }),
    });
    const data = await res.json();
    typingEl.remove();

    if (!res.ok) {
      renderMessage({ sender: "bot", text: data.error || "Something went wrong.", timestamp: new Date().toISOString() });
    } else {
      currentSessionId = data.session.id;
      renderMessage(data.bot_message);
      loadSessions();
    }
  } catch (err) {
    typingEl.remove();
    renderMessage({ sender: "bot", text: "Could not reach the server. Please check the backend is running.", timestamp: new Date().toISOString() });
  } finally {
    sendBtn.disabled = false;
    scrollToBottom();
  }
}

sendBtn.addEventListener("click", () => sendMessage(messageInput.value));
messageInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage(messageInput.value);
  }
});
messageInput.addEventListener("input", autoGrow);
function autoGrow() {
  messageInput.style.height = "auto";
  messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + "px";
}

suggestionChips.addEventListener("click", (e) => {
  if (e.target.classList.contains("chip")) {
    sendMessage(e.target.dataset.msg);
  }
});

// ------------------------------------------------------------------ //
// Rendering
// ------------------------------------------------------------------ //
function renderMessage(msg) {
  const row = document.createElement("div");
  row.className = `msg-row ${msg.sender}`;

  const avatar = document.createElement("div");
  avatar.className = "msg-avatar";
  avatar.textContent = msg.sender === "user" ? (currentUser.full_name.charAt(0).toUpperCase()) : "AI";

  const col = document.createElement("div");

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  bubble.textContent = msg.text;
  col.appendChild(bubble);

  if (msg.sender === "bot" && msg.intent) {
    const tag = document.createElement("div");
    tag.className = "intent-tag";
    tag.textContent = `intent: ${msg.intent}${msg.confidence ? " · " + Math.round(msg.confidence * 100) + "%" : ""}`;
    col.appendChild(tag);

    const fbRow = document.createElement("div");
    fbRow.className = "feedback-row";
    fbRow.innerHTML = `<button class="fb-btn" data-r="up">👍 Helpful</button><button class="fb-btn" data-r="down">👎 Not quite</button>`;
    fbRow.querySelectorAll(".fb-btn").forEach(btn => {
      btn.addEventListener("click", async () => {
        fbRow.querySelectorAll(".fb-btn").forEach(b => b.classList.remove("selected"));
        btn.classList.add("selected");
        await fetch(`${API_BASE}/feedback`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "include",
          body: JSON.stringify({ message_id: msg.id, rating: btn.dataset.r }),
        });
      });
    });
    col.appendChild(fbRow);
  }

  const meta = document.createElement("div");
  meta.className = "msg-meta";
  meta.textContent = formatTime(msg.timestamp);
  col.appendChild(meta);

  row.appendChild(avatar);
  row.appendChild(col);
  chatScroll.appendChild(row);
  return row;
}

function renderTyping() {
  const row = document.createElement("div");
  row.className = "msg-row bot";
  row.innerHTML = `
    <div class="msg-avatar">AI</div>
    <div class="msg-bubble"><div class="typing-indicator"><span></span><span></span><span></span></div></div>
  `;
  chatScroll.appendChild(row);
  return row;
}

function scrollToBottom() {
  chatScroll.scrollTop = chatScroll.scrollHeight;
}

function formatTime(iso) {
  try {
    return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch { return ""; }
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// ------------------------------------------------------------------ //
// Sidebar toggle (mobile) + logout
// ------------------------------------------------------------------ //
document.getElementById("menuToggle").addEventListener("click", () => {
  document.getElementById("sidebar").classList.toggle("open");
});

document.getElementById("logoutBtn").addEventListener("click", async () => {
  await fetch(`${API_BASE}/auth/logout`, { method: "POST", credentials: "include" });
  window.location.href = "/index.html";
});
