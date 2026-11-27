/* Progressive-enhancement script for clipboard copy, Explorer loading states, and tabular AI Chat UX. */
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

  function scopeLabel(scopeValue) {
    if (scopeValue === "clinical") return "Clinical Terminologies";
    if (scopeValue === "dicom") return "DICOM Standard";
    return "Clinical & DICOM";
  }

  function createSpinner(isLight) {
    var spinner = document.createElement("span");
    spinner.className = "loading-spinner loading-spinner-sm" + (isLight ? " loading-spinner-light" : "");
    spinner.setAttribute("aria-hidden", "true");
    return spinner;
  }

  function setButtonLoading(btn, isLoading, loadingLabel) {
    if (!btn) return;
    var original = btn.getAttribute("data-original-label");
    if (!original) {
      original = (btn.textContent || "").trim() || "Submit";
      btn.setAttribute("data-original-label", original);
    }

    btn.textContent = "";
    if (isLoading) {
      btn.disabled = true;
      btn.setAttribute("aria-busy", "true");
      btn.appendChild(createSpinner(btn.classList.contains("btn-primary")));
      var busySpan = document.createElement("span");
      busySpan.className = "btn-label";
      busySpan.textContent = loadingLabel || "Working\u2026";
      btn.appendChild(busySpan);
    } else {
      btn.disabled = false;
      btn.removeAttribute("aria-busy");
      var idleSpan = document.createElement("span");
      idleSpan.className = "btn-label";
      idleSpan.textContent = original;
      btn.appendChild(idleSpan);
    }
  }

  function createStageTicker(stages, onUpdate) {
    var timerIds = [];
    var stopped = false;

    if (Array.isArray(stages)) {
      stages.forEach(function (stage) {
        var delay = typeof stage.delayMs === "number" ? stage.delayMs : 0;
        if (delay <= 0) {
          onUpdate(stage.text);
        } else {
          var id = window.setTimeout(function () {
            if (!stopped) onUpdate(stage.text);
          }, delay);
          timerIds.push(id);
        }
      });
    }

    return {
      stop: function () {
        stopped = true;
        timerIds.forEach(function (id) {
          window.clearTimeout(id);
        });
        timerIds = [];
      }
    };
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

  // 2. Ontology Explorer (/explore) search & drill-down loading indicator
  function initExploreSearch() {
    var exploreForm = document.getElementById("explore-search-form");
    if (!exploreForm) return;

    var exploreBtn =
      document.getElementById("explore-search-btn") || exploreForm.querySelector('button[type="submit"]');
    var exploreQuery = document.getElementById("explore-query") || exploreForm.querySelector('input[name="q"]');
    var exploreScope = document.getElementById("explore-scope") || exploreForm.querySelector('select[name="scope"]');
    var exploreStatus = document.getElementById("explore-status");
    var exploreStatusText = document.getElementById("explore-status-text");

    var isSubmitting = false;
    var activeTicker = null;

    function resetExploreLoading() {
      if (activeTicker) {
        activeTicker.stop();
        activeTicker = null;
      }
      isSubmitting = false;
      setButtonLoading(exploreBtn, false);
      if (exploreStatus) exploreStatus.hidden = true;
      if (exploreStatusText) exploreStatusText.textContent = "";
    }

    function startExploreLoading(stages, btnLabel) {
      if (activeTicker) {
        activeTicker.stop();
        activeTicker = null;
      }
      setButtonLoading(exploreBtn, true, btnLabel);
      if (exploreStatus) exploreStatus.hidden = false;
      activeTicker = createStageTicker(stages, function (msg) {
        if (exploreStatusText) exploreStatusText.textContent = msg;
      });
    }

    exploreForm.addEventListener("submit", function (event) {
      if (isSubmitting) {
        event.preventDefault();
        return;
      }
      isSubmitting = true;

      var queryText = exploreQuery && exploreQuery.value ? exploreQuery.value.trim() : "";
      var scopeText = scopeLabel(exploreScope ? exploreScope.value : "auto");

      if (!queryText) {
        startExploreLoading([{ delayMs: 0, text: "Loading Ontology Explorer\u2026" }], "Loading\u2026");
      } else {
        startExploreLoading(
          [
            { delayMs: 0, text: "Parsing query with Clinical NLU\u2026" },
            { delayMs: 900, text: "Searching Neo4j " + scopeText + " graph & embeddings\u2026" },
            { delayMs: 2600, text: "Ranking matches and assembling results\u2026" }
          ],
          "Searching\u2026"
        );
      }
    });

    // Show loading status on same-tab drill-down navigation inside /explore
    document.addEventListener("click", function (event) {
      if (
        event.defaultPrevented ||
        event.button !== 0 ||
        event.metaKey ||
        event.ctrlKey ||
        event.shiftKey ||
        event.altKey
      ) {
        return;
      }
      var link = event.target.closest ? event.target.closest('a.drilldown-link, a[href^="/explore?"]') : null;
      if (!link || link.target === "_blank") return;
      if (isSubmitting) {
        event.preventDefault();
        return;
      }
      isSubmitting = true;
      startExploreLoading(
        [
          { delayMs: 0, text: "Inspecting entity in Neo4j\u2026" },
          { delayMs: 900, text: "Traversing 1-hop & 2-hop ontology neighborhood\u2026" }
        ],
        "Loading\u2026"
      );
    });

    // Reset state cleanly if restored from browser back/forward cache (bfcache)
    window.addEventListener("pageshow", resetExploreLoading);
  }

  // 3. Grounded Dictionary AI Chat enhancement (falls back to standard POST /chat without JS)
  function initChatForm() {
    var chatForm = document.getElementById("chat-form");
    var chatLog = document.getElementById("chat-log");
    var chatInput = document.getElementById("chat-message");
    var chatStatus = document.getElementById("chat-status");
    var clearBtn = document.getElementById("chat-clear-btn");
    var scopeSelect = document.getElementById("chat-scope");
    var submitBtn =
      document.getElementById("chat-submit-btn") || (chatForm ? chatForm.querySelector('button[type="submit"]') : null);

    if (!chatForm || !chatLog || !chatInput) return;

    var ctxCheckbox = chatForm.querySelector('input[name="include_context"]');
    var relCheckbox = chatForm.querySelector('input[name="include_related"]');
    var apiUrl = chatForm.getAttribute("data-api-url") || "/api/v1/chat";
    var history = [];
    var isSending = false;
    var activeChatTicker = null;

    function setChatStatus(message) {
      if (!chatStatus) return;
      chatStatus.textContent = "";
      if (!message) return;
      chatStatus.appendChild(createSpinner(false));
      var textSpan = document.createElement("span");
      textSpan.textContent = message;
      chatStatus.appendChild(textSpan);
    }

    function resetChatLoading() {
      if (activeChatTicker) {
        activeChatTicker.stop();
        activeChatTicker = null;
      }
      isSending = false;
      setButtonLoading(submitBtn, false);
      setChatStatus("");
    }

    window.addEventListener("pageshow", resetChatLoading);

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
      if (isSending) return;
      var text = chatInput.value.trim();
      if (!text) return;

      isSending = true;
      var useContext = ctxCheckbox ? Boolean(ctxCheckbox.checked) : true;
      var selectedScope = scopeSelect ? scopeSelect.value : "auto";
      var useRelated = relCheckbox ? Boolean(relCheckbox.checked) : false;

      appendMessage("user", text, {});
      chatInput.value = "";
      setButtonLoading(submitBtn, true, useContext ? "Searching\u2026" : "Thinking\u2026");

      var scopeText = scopeLabel(selectedScope);
      var stages;
      if (useContext) {
        var neo4jStage = useRelated
          ? "Searching Neo4j " + scopeText + " dictionary & traversing related concepts\u2026"
          : "Searching Neo4j " + scopeText + " dictionary\u2026";
        stages = [
          { delayMs: 0, text: "Parsing query with Clinical NLU\u2026" },
          { delayMs: 900, text: neo4jStage },
          { delayMs: 2400, text: "Generating grounded response with AI model\u2026" }
        ];
      } else {
        stages = [{ delayMs: 0, text: "Querying AI model\u2026" }];
      }

      if (activeChatTicker) activeChatTicker.stop();
      activeChatTicker = createStageTicker(stages, setChatStatus);

      var payload = {
        message: text,
        history: history.slice(-10),
        include_context: useContext,
        scope: selectedScope,
        include_related: useRelated
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
          resetChatLoading();
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
  }

  // 4. MCP Client Setup configurator on /keys (#mcp-setup)
  function initMcpSetup() {
    var section = document.getElementById("mcp-setup");
    if (!section) return;

    var mountPath = section.getAttribute("data-mount-path") || "/mcp";
    var sseMountPath = section.getAttribute("data-sse-mount-path") || "/mcp-sse";
    var defaultOrigin = section.getAttribute("data-default-origin") || window.location.origin;
    var placeholderKey = section.getAttribute("data-placeholder-key") || "<YOUR_GOSMART_API_KEY>";
    var serverAlias = section.getAttribute("data-server-alias") || "gosmart-dictionary";

    var hostInput = document.getElementById("mcp-server-host");
    var keyInput = document.getElementById("mcp-api-key-input");
    var endpointPreview = document.getElementById("mcp-endpoint-preview");
    var agentTitle = document.getElementById("mcp-agent-title");
    var locationsContainer = document.getElementById("mcp-config-locations");
    var mergeHintEl = document.getElementById("mcp-merge-hint");
    var filenameBadge = document.getElementById("mcp-filename-badge");
    var placeholderBadge = document.getElementById("mcp-placeholder-badge");
    var downloadBtn = document.getElementById("mcp-download-btn");
    var jsonCodeEl = document.getElementById("mcp-config-json");
    var cliWrap = document.getElementById("mcp-cli-wrap");
    var cliCommandEl = document.getElementById("mcp-cli-command");
    var stepsList = document.getElementById("mcp-setup-steps");
    var keyStatusEl = document.getElementById("mcp-key-status");

    var createKeyAgent = document.getElementById("create-key-mcp-agent");
    var createKeyHost = document.getElementById("create-key-mcp-host");
    var quickKeyName = document.getElementById("mcp-quick-key-name");
    var quickKeyAgent = document.getElementById("mcp-quick-key-agent");
    var quickKeyHost = document.getElementById("mcp-quick-key-host");

    var tabLinks = Array.prototype.slice.call(section.querySelectorAll("[data-mcp-agent]"));
    var activeAgent = "cursor";
    tabLinks.forEach(function (tab) {
      if (tab.classList.contains("active")) {
        activeAgent = tab.getAttribute("data-mcp-agent") || "cursor";
      }
    });
    var hadInitialServerKey = Boolean(keyInput && keyInput.value && keyInput.value.trim());

    function isLocalAuthority(authority) {
      var raw = (authority || "").trim().toLowerCase();
      var hostOnly;
      if (raw.charAt(0) === "[") {
        var closeIdx = raw.indexOf("]");
        hostOnly = closeIdx !== -1 ? raw.slice(1, closeIdx) : raw.slice(1);
      } else {
        hostOnly = raw.split(":")[0];
      }
      return (
        hostOnly === "localhost" ||
        hostOnly === "127.0.0.1" ||
        hostOnly === "::1" ||
        hostOnly === "0.0.0.0" ||
        hostOnly === "testserver" ||
        hostOnly.slice(-6) === ".local" ||
        hostOnly.slice(-10) === ".localhost"
      );
    }

    function normalizeEndpoint(rawHost) {
      var candidate = (rawHost || "").trim() || (defaultOrigin || "").trim() || window.location.origin;
      candidate = candidate.replace(/\/+$/, "");

      if (!/^https?:\/\//i.test(candidate)) {
        var authority = candidate.split("/")[0];
        var scheme = isLocalAuthority(authority) ? "http" : "https";
        candidate = scheme + "://" + candidate;
      }

      var cleanMount = "/" + String(mountPath || "/mcp").replace(/^\/+|\/+$/g, "");
      var cleanSse = "/" + String(sseMountPath || "/mcp-sse").replace(/^\/+|\/+$/g, "");

      try {
        var parsed = new URL(candidate);
        var cleanPath = parsed.pathname.replace(/\/+$/, "");
        var suffixes = [cleanSse + "/sse", cleanSse, cleanMount];
        for (var i = 0; i < suffixes.length; i += 1) {
          var sfx = suffixes[i];
          if (sfx && cleanPath.toLowerCase().slice(-sfx.length) === sfx.toLowerCase()) {
            cleanPath = cleanPath.slice(0, cleanPath.length - sfx.length).replace(/\/+$/, "");
            break;
          }
        }
        var baseOrigin = (parsed.protocol + "//" + parsed.host + cleanPath).replace(/\/+$/, "");
        return { base: baseOrigin, endpoint: baseOrigin + cleanMount };
      } catch (_err) {
        var fallbackBase = candidate.replace(/\/+$/, "");
        return { base: fallbackBase, endpoint: fallbackBase + cleanMount };
      }
    }

    function buildAgentSpec(agent, endpointUrl, effectiveKey) {
      var spec = {};
      var serverObj;

      if (agent === "claude-code") {
        spec.label = "Claude Code";
        spec.filename = ".mcp.json";
        serverObj = {
          type: "http",
          url: endpointUrl,
          headers: { "X-API-Key": effectiveKey }
        };
        spec.locations = [
          { scope: "Project (shared / repo root)", path: ".mcp.json" },
          { scope: "Global (macOS / Linux)", path: "~/.claude.json" },
          { scope: "Global (Windows)", path: "%USERPROFILE%\\.claude.json" }
        ];
        spec.mergeHint =
          'Save as .mcp.json in your project root, or merge the "gosmart-dictionary" entry inside the existing "mcpServers" object in ~/.claude.json (or run the CLI command below).';
        spec.cli =
          "claude mcp add --transport http " +
          serverAlias +
          " " +
          endpointUrl +
          ' --header "X-API-Key: ' +
          effectiveKey +
          '"';
        spec.steps = [
          "Place or merge the JSON into .mcp.json (project) or ~/.claude.json (global), or run the claude mcp add command below.",
          'Keep "type": "http" intact\u2014Claude Code requires an explicit transport type for remote HTTP MCP servers.',
          "Run /mcp inside an interactive Claude Code session to verify gosmart-dictionary is connected."
        ];
      } else if (agent === "claude-desktop") {
        spec.label = "Claude Desktop";
        spec.filename = "claude_desktop_config.json";
        serverObj = {
          command: "npx",
          args: ["-y", "mcp-remote", endpointUrl, "--header", "X-API-Key:${GOSMART_API_KEY}"],
          env: { GOSMART_API_KEY: effectiveKey }
        };
        spec.locations = [
          { scope: "macOS", path: "~/Library/Application Support/Claude/claude_desktop_config.json" },
          { scope: "Windows", path: "%APPDATA%\\Claude\\claude_desktop_config.json" },
          { scope: "Linux", path: "~/.config/Claude/claude_desktop_config.json" }
        ];
        spec.mergeHint =
          'Open via Claude Desktop \u2192 Settings \u2192 Developer \u2192 Edit Config, and merge "gosmart-dictionary" under "mcpServers".';
        spec.cli = "";
        spec.steps = [
          "Ensure Node.js 18+ (npx) is installed\u2014claude_desktop_config.json bridges remote HTTP headers via mcp-remote.",
          "Save or merge the JSON block into claude_desktop_config.json at your OS path above.",
          "Restart Claude Desktop completely to load the GoSmart Dictionary tools."
        ];
      } else if (agent === "gemini") {
        spec.label = "Gemini CLI";
        spec.filename = "settings.json";
        serverObj = {
          httpUrl: endpointUrl,
          headers: { "X-API-Key": effectiveKey }
        };
        spec.locations = [
          { scope: "Project (workspace)", path: ".gemini/settings.json" },
          { scope: "Global (macOS / Linux)", path: "~/.gemini/settings.json" },
          { scope: "Global (Windows)", path: "%USERPROFILE%\\.gemini\\settings.json" }
        ];
        spec.mergeHint =
          'settings.json also stores general Gemini CLI settings\u2014merge "gosmart-dictionary" inside its "mcpServers" object rather than overwriting an existing settings.json (or run the CLI command below).';
        spec.cli =
          'gemini mcp add --transport http --header "X-API-Key: ' +
          effectiveKey +
          '" ' +
          serverAlias +
          " " +
          endpointUrl;
        spec.steps = [
          "Merge the JSON block into ~/.gemini/settings.json or .gemini/settings.json, or run the gemini mcp add command below.",
          'Keep "httpUrl" (Gemini CLI reserves "url" for legacy SSE) and the hyphenated alias "gosmart-dictionary".',
          "Run /mcp inside an interactive gemini session to confirm gosmart-dictionary is active."
        ];
      } else if (agent === "antigravity") {
        spec.label = "Antigravity";
        spec.filename = "mcp_config.json";
        serverObj = {
          serverUrl: endpointUrl,
          headers: { "X-API-Key": effectiveKey }
        };
        spec.locations = [
          { scope: "Workspace", path: ".agents/mcp_config.json" },
          { scope: "Global (macOS / Linux)", path: "~/.gemini/config/mcp_config.json" },
          { scope: "Global (Windows)", path: "%USERPROFILE%\\.gemini\\config\\mcp_config.json" }
        ];
        spec.mergeHint =
          'Save directly as .agents/mcp_config.json for a single workspace, or merge "gosmart-dictionary" under "mcpServers" in ~/.gemini/config/mcp_config.json (also accessible via MCP Servers \u2192 Manage MCP Servers \u2192 View raw config).';
        spec.cli = "";
        spec.steps = [
          "Open MCP Servers \u2192 Manage MCP Servers \u2192 View raw config in Antigravity, or open ~/.gemini/config/mcp_config.json (or <workspace>/.agents/mcp_config.json).",
          'Merge "gosmart-dictionary" under "mcpServers" and save (Antigravity uses "serverUrl" and does not accept Gemini CLI\'s "httpUrl").',
          "Refresh MCP servers in Antigravity to load the 15 GoSmart Dictionary tools."
        ];
      } else {
        spec.label = "Cursor";
        spec.filename = "mcp.json";
        serverObj = {
          url: endpointUrl,
          headers: { "X-API-Key": effectiveKey }
        };
        spec.locations = [
          { scope: "Project (workspace)", path: ".cursor/mcp.json" },
          { scope: "Global (macOS / Linux)", path: "~/.cursor/mcp.json" },
          { scope: "Global (Windows)", path: "%USERPROFILE%\\.cursor\\mcp.json" }
        ];
        spec.mergeHint =
          'Save directly as a new mcp.json file, or merge "gosmart-dictionary" inside your existing "mcpServers" object.';
        spec.cli = "";
        spec.steps = [
          "Save or merge the JSON into .cursor/mcp.json (project) or ~/.cursor/mcp.json (global), or open Cursor Settings \u2192 Tools & MCP.",
          "Save the file and confirm gosmart-dictionary shows a green status indicator.",
          'Ask Cursor a terminology or DICOM question (e.g. "Search SNOMED for acute pneumonia").'
        ];
      }

      var root = { mcpServers: {} };
      root.mcpServers[serverAlias] = serverObj;
      spec.json = JSON.stringify(root, null, 2);
      return spec;
    }

    function renderLocations(locations) {
      if (!locationsContainer) return;
      locationsContainer.textContent = "";
      locations.forEach(function (loc) {
        var row = document.createElement("div");
        row.className = "mcp-location-row";

        var scopeSpan = document.createElement("span");
        scopeSpan.className = "mcp-location-scope";
        scopeSpan.textContent = loc.scope;

        var pathCode = document.createElement("code");
        pathCode.className = "mono mcp-location-path";
        pathCode.textContent = loc.path;

        var copyBtn = document.createElement("button");
        copyBtn.type = "button";
        copyBtn.className = "btn btn-secondary btn-xs";
        copyBtn.setAttribute("data-copy-text", loc.path);
        copyBtn.textContent = "Copy Path";

        row.appendChild(scopeSpan);
        row.appendChild(pathCode);
        row.appendChild(copyBtn);
        locationsContainer.appendChild(row);
      });
    }

    function renderSteps(steps) {
      if (!stepsList) return;
      stepsList.textContent = "";
      steps.forEach(function (stepText) {
        var li = document.createElement("li");
        li.textContent = stepText;
        stepsList.appendChild(li);
      });
    }

    function updateDownloadButtonLabel(filename) {
      if (!downloadBtn) return;
      downloadBtn.setAttribute("download", filename);
      downloadBtn.setAttribute("data-download-filename", filename);
      downloadBtn.textContent = "";
      downloadBtn.appendChild(document.createTextNode("Download "));
      var span = document.createElement("span");
      span.className = "mono";
      span.id = "mcp-download-label";
      span.textContent = filename;
      downloadBtn.appendChild(span);
    }

    function renderMcpConfigurator(userEditedKey) {
      var rawHost = hostInput ? hostInput.value : "";
      var resolved = normalizeEndpoint(rawHost);
      var hostValue = (rawHost || "").trim() || resolved.base;
      var rawKey = keyInput && keyInput.value ? keyInput.value.trim() : "";
      var hasRealKey = Boolean(rawKey);
      var effectiveKey = hasRealKey ? rawKey : placeholderKey;
      var spec = buildAgentSpec(activeAgent, resolved.endpoint, effectiveKey);

      tabLinks.forEach(function (tab) {
        var tabAgent = tab.getAttribute("data-mcp-agent") || "cursor";
        var isSelected = tabAgent === activeAgent;
        tab.classList.toggle("active", isSelected);
        tab.href = "/keys?agent=" + encodeURIComponent(tabAgent) + "&mcp_host=" + encodeURIComponent(hostValue) + "#mcp-setup";
      });

      if (endpointPreview) endpointPreview.textContent = resolved.endpoint;
      if (agentTitle) agentTitle.textContent = spec.label;
      renderLocations(spec.locations);
      if (mergeHintEl) mergeHintEl.textContent = spec.mergeHint;
      if (filenameBadge) filenameBadge.textContent = spec.filename;
      if (placeholderBadge) placeholderBadge.classList.toggle("is-hidden", hasRealKey);
      if (jsonCodeEl) jsonCodeEl.textContent = spec.json;

      if (downloadBtn) {
        downloadBtn.href =
          "/keys/mcp-config.json?agent=" + encodeURIComponent(activeAgent) + "&mcp_host=" + encodeURIComponent(hostValue);
        updateDownloadButtonLabel(spec.filename);
      }

      if (cliWrap && cliCommandEl) {
        if (spec.cli) {
          cliWrap.classList.remove("is-hidden");
          cliCommandEl.textContent = spec.cli;
        } else {
          cliWrap.classList.add("is-hidden");
          cliCommandEl.textContent = "";
        }
      }

      renderSteps(spec.steps);

      if (createKeyAgent) createKeyAgent.value = activeAgent;
      if (createKeyHost) createKeyHost.value = hostValue;
      if (quickKeyAgent) quickKeyAgent.value = activeAgent;
      if (quickKeyHost) quickKeyHost.value = hostValue;
      if (quickKeyName) quickKeyName.value = "MCP \u2014 " + spec.label;

      if (userEditedKey && keyStatusEl && !hadInitialServerKey) {
        if (hasRealKey) {
          keyStatusEl.textContent = "Injected your API key into the JSON snippet (kept only in browser memory).";
        } else {
          keyStatusEl.textContent =
            "Existing keys store only SHA-256 hashes. Paste a saved gs_dict_... key above or click Generate Key.";
        }
      }
    }

    tabLinks.forEach(function (tab) {
      tab.addEventListener("click", function (event) {
        if (
          event.defaultPrevented ||
          event.button !== 0 ||
          event.metaKey ||
          event.ctrlKey ||
          event.shiftKey ||
          event.altKey
        ) {
          return;
        }
        event.preventDefault();
        activeAgent = tab.getAttribute("data-mcp-agent") || "cursor";
        renderMcpConfigurator(false);
      });
    });

    if (hostInput) {
      hostInput.addEventListener("input", function () {
        renderMcpConfigurator(false);
      });
    }

    if (keyInput) {
      keyInput.addEventListener("input", function () {
        hadInitialServerKey = false;
        renderMcpConfigurator(true);
      });
    }

    if (downloadBtn) {
      downloadBtn.addEventListener("click", function (event) {
        if (!window.Blob || !window.URL || !window.URL.createObjectURL || !jsonCodeEl) return;
        event.preventDefault();
        var filename = downloadBtn.getAttribute("data-download-filename") || "mcp.json";
        var content = (jsonCodeEl.textContent || "").trim() + "\n";
        var blob = new window.Blob([content], { type: "application/json;charset=utf-8" });
        var blobUrl = window.URL.createObjectURL(blob);
        var tempLink = document.createElement("a");
        tempLink.href = blobUrl;
        tempLink.download = filename;
        document.body.appendChild(tempLink);
        tempLink.click();
        document.body.removeChild(tempLink);
        window.setTimeout(function () {
          window.URL.revokeObjectURL(blobUrl);
        }, 1000);

        downloadBtn.textContent = "Downloaded!";
        window.setTimeout(function () {
          updateDownloadButtonLabel(filename);
        }, 1600);
      });
    }
  }

  initExploreSearch();
  initChatForm();
  initMcpSetup();
})();
