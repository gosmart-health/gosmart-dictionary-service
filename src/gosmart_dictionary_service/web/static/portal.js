/* Small progressive-enhancement script for API key copy and AI Chat UX. */
(function () {
  "use strict";

  // 1. Copy-to-clipboard buttons (e.g., newly generated API key)
  document.querySelectorAll("[data-copy-target]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var targetId = btn.getAttribute("data-copy-target");
      var el = targetId ? document.getElementById(targetId) : null;
      if (!el || !navigator.clipboard) return;
      var text = el.textContent ? el.textContent.trim() : "";
      if (!text) return;
      navigator.clipboard.writeText(text).then(function () {
        var original = btn.textContent;
        btn.textContent = "Copied!";
        window.setTimeout(function () {
          btn.textContent = original;
        }, 1800);
      });
    });
  });

  // 2. Grounded Dictionary AI Chat enhancement (falls back to standard POST /chat without JS)
  var chatForm = document.getElementById("chat-form");
  var chatLog = document.getElementById("chat-log");
  var chatInput = document.getElementById("chat-message");
  var chatStatus = document.getElementById("chat-status");
  var clearBtn = document.getElementById("chat-clear-btn");
  var ctxCheckbox = document.getElementById("chat-include-context");

  if (!chatForm || !chatLog || !chatInput) return;

  var apiUrl = chatForm.getAttribute("data-api-url") || "/api/v1/chat";
  var history = [];

  function appendMessage(role, content, citations, metaText) {
    var emptyState = document.getElementById("chat-empty-state");
    if (emptyState) emptyState.remove();

    var wrapper = document.createElement("div");
    wrapper.className = "chat-msg " + (role === "user" ? "chat-msg-user" : "chat-msg-assistant");

    var body = document.createElement("div");
    body.textContent = content;
    wrapper.appendChild(body);

    if (Array.isArray(citations) && citations.length > 0) {
      var citList = document.createElement("div");
      citList.className = "citation-list";
      citations.forEach(function (c) {
        var badge = document.createElement("span");
        badge.className = "citation-badge";
        var parts = [c.identifier, c.keyword || c.name];
        if (c.vr) parts.push("VR=" + c.vr);
        badge.textContent = parts.filter(Boolean).join(" · ");
        if (c.snippet) badge.title = c.snippet;
        citList.appendChild(badge);
      });
      wrapper.appendChild(citList);
    }

    if (metaText) {
      var meta = document.createElement("div");
      meta.className = "chat-msg-meta";
      meta.textContent = metaText;
      wrapper.appendChild(meta);
    }

    chatLog.appendChild(wrapper);
    chatLog.scrollTop = chatLog.scrollHeight;
  }

  chatForm.addEventListener("submit", function (event) {
    event.preventDefault();
    var text = chatInput.value.trim();
    if (!text) return;

    var submitBtn = chatForm.querySelector('button[type="submit"]');
    appendMessage("user", text, [], null);
    chatInput.value = "";
    if (submitBtn) submitBtn.disabled = true;
    if (chatStatus) chatStatus.textContent = "Searching dictionary & querying model...";

    var payload = {
      message: text,
      history: history.slice(-10),
      include_context: ctxCheckbox ? Boolean(ctxCheckbox.checked) : true
    };

    window
      .fetch(apiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        credentials: "same-origin",
        body: JSON.stringify(payload)
      })
      .then(function (resp) {
        return resp.json().then(function (data) {
          return { ok: resp.ok, status: resp.status, data: data };
        });
      })
      .then(function (res) {
        if (!res.ok) {
          var detail = (res.data && res.data.detail) || "Request failed (HTTP " + res.status + ")";
          appendMessage("assistant", "Error: " + detail, [], null);
          return;
        }
        var reply = res.data.reply || "";
        var citations = res.data.citations || [];
        var meta = (res.data.provider || "") + "/" + (res.data.model || "");
        appendMessage("assistant", reply, citations, meta);
        history.push({ role: "user", content: text });
        history.push({ role: "assistant", content: reply });
      })
      .catch(function (err) {
        appendMessage("assistant", "Network error: " + String(err), [], null);
      })
      .finally(function () {
        if (submitBtn) submitBtn.disabled = false;
        if (chatStatus) chatStatus.textContent = "";
        chatInput.focus();
      });
  });

  if (clearBtn) {
    clearBtn.addEventListener("click", function () {
      history = [];
      chatLog.innerHTML =
        '<p id="chat-empty-state" class="muted">Conversation cleared. Ask a DICOM or clinical terminology question below.</p>';
      chatInput.focus();
    });
  }
})();
