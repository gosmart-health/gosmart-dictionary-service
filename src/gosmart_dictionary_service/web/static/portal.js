/* Progressive-enhancement script for clipboard copy, tabular AI Chat UX, and quick drill-down prompts. */
(function () {
  "use strict";

  function copyWithFeedback(btn, text) {
    if (!text || !navigator.clipboard) return;
    navigator.clipboard.writeText(text).then(function () {
      var original = btn.textContent;
      btn.textContent = "Copied!";
      window.setTimeout(function () {
        btn.textContent = original;
      }, 1600);
    });
  }

  // 1. Delegated copy handlers ([data-copy-target] & [data-copy-text])
  document.addEventListener("click", function (event) {
    var targetBtn = event.target.closest ? event.target.closest("[data-copy-target]") : null;
    if (targetBtn) {
      var targetId = targetBtn.getAttribute("data-copy-target");
      var el = targetId ? document.getElementById(targetId) : null;
      var text = el && el.textContent ? el.textContent.trim() : "";
      copyWithFeedback(targetBtn, text);
      return;
    }

    var inlineBtn = event.target.closest ? event.target.closest("[data-copy-text]") : null;
    if (inlineBtn) {
      var raw = inlineBtn.getAttribute("data-copy-text") || "";
      copyWithFeedback(inlineBtn, raw.trim());
    }
  });

  // 2. Grounded Dictionary AI Chat enhancement (falls back to standard POST /chat without JS)
  var chatForm = document.getElementById("chat-form");
  var chatLog = document.getElementById("chat-log");
  var chatInput = document.getElementById("chat-message");
  var chatStatus = document.getElementById("chat-status");
  var clearBtn = document.getElementById("chat-clear-btn");
  var scopeSelect = document.getElementById("chat-scope");

  if (!chatForm || !chatLog || !chatInput) return;

  var ctxCheckbox = chatForm.querySelector('input[name="include_context"]');
  var relCheckbox = chatForm.querySelector('input[name="include_related"]');
  var apiUrl = chatForm.getAttribute("data-api-url") || "/api/v1/chat";
  var history = [];

  // Collapse older server-rendered citation drawers if multiple messages exist
  chatLog.querySelectorAll('details.citation-drawer[data-is-last="0"]').forEach(function (d) {
    d.removeAttribute("open");
  });

  function syncContextControls() {
    var enabled = ctxCheckbox ? Boolean(ctxCheckbox.checked) : true;
    if (scopeSelect) scopeSelect.disabled = !enabled;
    if (relCheckbox) relCheckbox.disabled = !enabled;
  }

  if (ctxCheckbox) {
    ctxCheckbox.addEventListener("change", syncContextControls);
    syncContextControls();
  }

  // Delegated click handler for quick-prompt chips & row-level "Ask in Chat" buttons
  document.addEventListener("click", function (event) {
    var promptBtn = event.target.closest ? event.target.closest("[data-chat-prompt]") : null;
    if (!promptBtn) return;
    var promptText = promptBtn.getAttribute("data-chat-prompt") || "";
    if (!promptText) return;
    chatInput.value = promptText;
    if (relCheckbox && promptText.toLowerCase().indexOf("related") !== -1) {
      relCheckbox.checked = true;
    }
    chatInput.focus();
  });

  function collapsePreviousCitationDrawers() {
    chatLog.querySelectorAll("details.citation-drawer[open]").forEach(function (d) {
      d.removeAttribute("open");
    });
  }

  function buildNluBar(nlu, durationMs) {
    if (!nlu) return null;
    var bar = document.createElement("div");
    bar.className = "chat-nlu-bar";

    var intentPill = document.createElement("span");
    intentPill.className = "pill pill-info mono";
    intentPill.textContent = String(nlu.intent || "search")
      .replace(/_/g, " ")
      .toUpperCase();
    bar.appendChild(intentPill);

    if (Array.isArray(nlu.extracted_codes) && nlu.extracted_codes.length > 0) {
      var codesSpan = document.createElement("span");
      codesSpan.textContent = "Codes: ";
      var codesStrong = document.createElement("strong");
      codesStrong.className = "mono";
      codesStrong.textContent = nlu.extracted_codes.join(", ");
      codesSpan.appendChild(codesStrong);
      bar.appendChild(codesSpan);
    }

    if (nlu.rewritten_query) {
      var rwSpan = document.createElement("span");
      rwSpan.textContent = "Rewritten: ";
      var rwEm = document.createElement("em");
      rwEm.textContent = '"' + nlu.rewritten_query + '"';
      rwSpan.appendChild(rwEm);
      bar.appendChild(rwSpan);
    }

    if (Array.isArray(nlu.target_vocabularies) && nlu.target_vocabularies.length > 0) {
      var targetSpan = document.createElement("span");
      targetSpan.textContent = "Target: " + nlu.target_vocabularies.join(", ");
      bar.appendChild(targetSpan);
    }

    if (typeof durationMs === "number") {
      var msSpan = document.createElement("span");
      msSpan.className = "mono";
      msSpan.textContent = durationMs.toFixed(1) + " ms";
      bar.appendChild(msSpan);
    }

    return bar;
  }

  function buildWarningsBox(warnings) {
    if (!Array.isArray(warnings) || warnings.length === 0) return null;
    var box = document.createElement("div");
    box.className = "chat-warning-box";
    box.setAttribute("role", "note");
    warnings.forEach(function (w) {
      var line = document.createElement("div");
      line.textContent = "\u26A0 " + String(w);
      box.appendChild(line);
    });
    return box;
  }

  function buildCitationDrawer(citations) {
    if (!Array.isArray(citations) || citations.length === 0) return null;

    var details = document.createElement("details");
    details.className = "citation-drawer";
    details.setAttribute("open", "open");

    var summary = document.createElement("summary");
    summary.className = "citation-drawer-summary";
    var titleSpan = document.createElement("span");
    titleSpan.textContent = "Grounded Dictionary Results (" + citations.length + ")";
    var hintSpan = document.createElement("span");
    hintSpan.className = "muted";
    hintSpan.textContent = "Click any code/tag to drill down";
    summary.appendChild(titleSpan);
    summary.appendChild(hintSpan);
    details.appendChild(summary);

    var tableWrap = document.createElement("div");
    tableWrap.className = "table-wrap citation-table-wrap";

    var table = document.createElement("table");
    table.className = "citation-table";

    var thead = document.createElement("thead");
    var headRow = document.createElement("tr");
    ["Schema / Type", "Code / Tag", "Display / Name", "Details", "Score", "Actions"].forEach(function (col) {
      var th = document.createElement("th");
      th.textContent = col;
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);

    var tbody = document.createElement("tbody");
    citations.forEach(function (c) {
      var tr = document.createElement("tr");

      // 1. Schema / Type
      var tdSchema = document.createElement("td");
      var schemaPill = document.createElement("span");
      schemaPill.className = "pill pill-schema";
      schemaPill.textContent = c.vocabulary || String(c.entity_type || "").replace(/^Dicom/, "DICOM ");
      tdSchema.appendChild(schemaPill);
      tr.appendChild(tdSchema);

      // 2. Code / Tag (drill-down link)
      var tdCode = document.createElement("td");
      tdCode.className = "mono";
      if (c.drilldown_url) {
        var codeLink = document.createElement("a");
        codeLink.href = c.drilldown_url;
        codeLink.className = "drilldown-link";
        codeLink.textContent = c.identifier || c.code || "";
        tdCode.appendChild(codeLink);
      } else {
        tdCode.textContent = c.identifier || c.code || "";
      }
      tr.appendChild(tdCode);

      // 3. Display / Name + snippet + related chips
      var tdName = document.createElement("td");
      var nameHeader = document.createElement("div");
      var nameStrong = document.createElement("strong");
      nameStrong.textContent = c.name || "";
      nameHeader.appendChild(nameStrong);
      if (c.retired) {
        nameHeader.appendChild(document.createTextNode(" "));
        var retPill = document.createElement("span");
        retPill.className = "pill pill-revoked";
        retPill.textContent = "Retired";
        nameHeader.appendChild(retPill);
      }
      tdName.appendChild(nameHeader);

      if (c.snippet) {
        var sub = document.createElement("div");
        sub.className = "citation-subtext";
        sub.textContent = c.snippet;
        tdName.appendChild(sub);
      }

      if (Array.isArray(c.related) && c.related.length > 0) {
        var relWrap = document.createElement("div");
        relWrap.className = "citation-related-inline";
        var relLbl = document.createElement("span");
        relLbl.className = "muted";
        relLbl.textContent = "Related:";
        relWrap.appendChild(relLbl);
        c.related.slice(0, 5).forEach(function (rel) {
          var relLink = document.createElement("a");
          relLink.href = rel.drilldown_url || "/explore";
          relLink.className = "citation-rel-chip";
          relLink.title = (rel.relationship_id || rel.relation || "") + ": " + (rel.display || "");
          relLink.textContent = (rel.relationship_id || rel.relation || "rel") + " \u2192 " + (rel.identifier || "");
          relWrap.appendChild(relLink);
        });
        tdName.appendChild(relWrap);
      }
      tr.appendChild(tdName);

      // 4. Details
      var tdDetails = document.createElement("td");
      tdDetails.className = "muted";
      if (c.entity_type === "Concept") {
        if (c.domain) {
          var domDiv = document.createElement("div");
          domDiv.textContent = "Domain: " + c.domain;
          tdDetails.appendChild(domDiv);
        }
        if (c.standard_concept === "S") {
          var stdDiv = document.createElement("div");
          stdDiv.textContent = "Standard (S)";
          tdDetails.appendChild(stdDiv);
        }
      } else if (c.entity_type === "DicomElement") {
        if (c.keyword) {
          var kwDiv = document.createElement("div");
          kwDiv.className = "mono";
          kwDiv.textContent = c.keyword;
          tdDetails.appendChild(kwDiv);
        }
        var vrBits = [];
        if (c.vr) vrBits.push("VR=" + c.vr);
        if (c.vm) vrBits.push("VM=" + c.vm);
        if (vrBits.length > 0) {
          var vrDiv = document.createElement("div");
          vrDiv.textContent = vrBits.join(" \u00B7 ");
          tdDetails.appendChild(vrDiv);
        }
      } else {
        tdDetails.textContent = c.entity_type || "";
      }
      tr.appendChild(tdDetails);

      // 5. Score
      var tdScore = document.createElement("td");
      tdScore.className = "mono";
      tdScore.textContent = typeof c.score === "number" ? c.score.toFixed(2) : "\u2014";
      if (c.source) {
        var srcDiv = document.createElement("div");
        srcDiv.className = "muted";
        srcDiv.style.fontSize = "0.74rem";
        srcDiv.textContent = c.source;
        tdScore.appendChild(srcDiv);
      }
      tr.appendChild(tdScore);

      // 6. Actions (Drill Down + Ask in Chat + Copy Macro)
      var tdActions = document.createElement("td");
      var actWrap = document.createElement("div");
      actWrap.className = "citation-actions";

      if (c.drilldown_url) {
        var ddLink = document.createElement("a");
        ddLink.href = c.drilldown_url;
        ddLink.className = "btn btn-secondary btn-xs";
        ddLink.textContent = "Drill Down";
        actWrap.appendChild(ddLink);
      }

      var askBtn = document.createElement("button");
      askBtn.type = "button";
      askBtn.className = "btn btn-secondary btn-xs";
      askBtn.setAttribute(
        "data-chat-prompt",
        "Show related concepts and crosswalks for " + (c.identifier || "") + " (" + (c.name || "") + ")"
      );
      askBtn.textContent = "Ask in Chat";
      actWrap.appendChild(askBtn);

      if (c.macro) {
        var copyBtn = document.createElement("button");
        copyBtn.type = "button";
        copyBtn.className = "btn btn-secondary btn-xs";
        copyBtn.setAttribute("data-copy-text", c.macro);
        copyBtn.title = c.macro;
        copyBtn.textContent = "Copy Macro";
        actWrap.appendChild(copyBtn);
      }

      tdActions.appendChild(actWrap);
      tr.appendChild(tdActions);

      tbody.appendChild(tr);
    });

    table.appendChild(tbody);
    tableWrap.appendChild(table);
    details.appendChild(tableWrap);
    return details;
  }

  function appendMessage(role, content, extra) {
    var emptyState = document.getElementById("chat-empty-state");
    if (emptyState) emptyState.remove();

    if (role === "assistant") {
      collapsePreviousCitationDrawers();
    }

    var opts = extra || {};
    var wrapper = document.createElement("div");
    wrapper.className = "chat-msg " + (role === "user" ? "chat-msg-user" : "chat-msg-assistant");

    var body = document.createElement("div");
    body.className = "chat-msg-body";
    body.textContent = content;
    wrapper.appendChild(body);

    var nluBar = buildNluBar(opts.nlu, opts.searchDurationMs);
    if (nluBar) wrapper.appendChild(nluBar);

    var warnBox = buildWarningsBox(opts.warnings);
    if (warnBox) wrapper.appendChild(warnBox);

    var drawer = buildCitationDrawer(opts.citations);
    if (drawer) wrapper.appendChild(drawer);

    if (opts.metaText) {
      var meta = document.createElement("div");
      meta.className = "chat-msg-meta";
      meta.textContent = opts.metaText;
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
    appendMessage("user", text, {});
    chatInput.value = "";
    if (submitBtn) submitBtn.disabled = true;
    if (chatStatus) chatStatus.textContent = "Running NLU search & querying model...";

    var payload = {
      message: text,
      history: history.slice(-10),
      include_context: ctxCheckbox ? Boolean(ctxCheckbox.checked) : true,
      scope: scopeSelect ? scopeSelect.value : "auto",
      include_related: relCheckbox ? Boolean(relCheckbox.checked) : false
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
          appendMessage("assistant", "Error: " + detail, {});
          return;
        }
        var reply = res.data.reply || "";
        var meta = (res.data.provider || "") + "/" + (res.data.model || "");
        appendMessage("assistant", reply, {
          citations: res.data.citations || [],
          nlu: res.data.nlu || null,
          warnings: res.data.warnings || [],
          searchDurationMs: res.data.search_duration_ms,
          metaText: meta
        });
        history.push({ role: "user", content: text });
        history.push({ role: "assistant", content: reply });
      })
      .catch(function (err) {
        appendMessage("assistant", "Network error: " + String(err), {});
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
      chatLog.textContent = "";
      var p = document.createElement("p");
      p.id = "chat-empty-state";
      p.className = "muted";
      p.textContent = "Conversation cleared. Ask a clinical terminology or DICOM question below.";
      chatLog.appendChild(p);
      chatInput.focus();
    });
  }
})();
