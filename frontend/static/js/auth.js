const API_BASE = "/api";

// ---- Tab switching -----------------------------------------------------
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".auth-form").forEach(f => f.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.tab + "Form").classList.add("active");
  });
});

// ---- Already logged in? redirect straight to chat ----------------------
(async function checkSession() {
  try {
    const res = await fetch(`${API_BASE}/auth/me`, { credentials: "include" });
    const data = await res.json();
    if (data.user) {
      window.location.href = "/chat.html";
    }
  } catch (e) { /* backend not reachable yet — stay on auth page */ }
})();

// ---- Login ---------------------------------------------------------------
document.getElementById("loginForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const errorEl = document.getElementById("loginError");
  errorEl.textContent = "";

  const email = document.getElementById("loginEmail").value.trim();
  const password = document.getElementById("loginPassword").value;

  try {
    const res = await fetch(`${API_BASE}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ email, password }),
    });
    const data = await res.json();
    if (!res.ok) {
      errorEl.textContent = data.error || "Login failed";
      return;
    }
    window.location.href = "/chat.html";
  } catch (err) {
    errorEl.textContent = "Could not reach the server. Is the Flask backend running?";
  }
});

// ---- Signup ---------------------------------------------------------------
document.getElementById("signupForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const errorEl = document.getElementById("signupError");
  errorEl.textContent = "";

  const full_name = document.getElementById("signupName").value.trim();
  const email = document.getElementById("signupEmail").value.trim();
  const password = document.getElementById("signupPassword").value;
  const role = document.getElementById("signupRole").value;
  const stream = document.getElementById("signupStream").value;

  try {
    const res = await fetch(`${API_BASE}/auth/signup`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ full_name, email, password, role, stream }),
    });
    const data = await res.json();
    if (!res.ok) {
      errorEl.textContent = data.error || "Signup failed";
      return;
    }
    window.location.href = "/chat.html";
  } catch (err) {
    errorEl.textContent = "Could not reach the server. Is the Flask backend running?";
  }
});
