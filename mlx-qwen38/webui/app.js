(function () {
  "use strict";

  const STORAGE_KEYS = {
    CONVERSATIONS: "moe_webui_conversations_v1",
    ACTIVE_CONV_ID: "moe_webui_active_conv_v1",
    SETTINGS: "moe_webui_settings_v1",
    MODE_THINKING: "moe_webui_thinking_v1",
    PERF_HISTORY: "moe_webui_perf_history_v1",
    THEME: "moe_webui_theme_v1",
  };

  const DEFAULT_SETTINGS = {
    systemPrompt: "",
    maxTokens: 1024,
    temperature: 0.7,
    topP: 0.9,
    seed: "",
    stop: "",
  };

  const MAX_SPARK_POINTS = 30;

  // State
  let conversations = [];
  let activeConvId = null;
  let settings = { ...DEFAULT_SETTINGS };
  let enableThinking = true;
  let perfHistory = []; // [{ prefillTps, decodeTps, ts }]
  let activeAbortController = null;
  let isGenerating = false;
  let statusPollTimer = null;
  let lastRecordedFinishedAt = null;
  let modalCallback = null;

  // DOM Elements
  const el = {
    offlineBanner: document.getElementById("offlineBanner"),
    offlineText: document.getElementById("offlineText"),
    retryConnBtn: document.getElementById("retryConnBtn"),
    toggleSidebarBtn: document.getElementById("toggleSidebarBtn"),
    modelSelect: document.getElementById("modelSelect"),
    modeChatBtn: document.getElementById("modeChatBtn"),
    modeThinkBtn: document.getElementById("modeThinkBtn"),
    liveDecodeTps: document.getElementById("liveDecodeTps"),
    busyIndicator: document.getElementById("busyIndicator"),
    busyText: document.getElementById("busyText"),
    toggleSettingsBtn: document.getElementById("toggleSettingsBtn"),
    togglePerfBtn: document.getElementById("togglePerfBtn"),
    themeToggleBtn: document.getElementById("themeToggleBtn"),
    convSidebar: document.getElementById("convSidebar"),
    newConvBtn: document.getElementById("newConvBtn"),
    convList: document.getElementById("convList"),
    exportJsonBtn: document.getElementById("exportJsonBtn"),
    exportMdBtn: document.getElementById("exportMdBtn"),
    clearChatBtn: document.getElementById("clearChatBtn"),
    inlineErrorBox: document.getElementById("inlineErrorBox"),
    inlineErrorMsg: document.getElementById("inlineErrorMsg"),
    dismissErrorBtn: document.getElementById("dismissErrorBtn"),
    messagesContainer: document.getElementById("messagesContainer"),
    emptyState: document.getElementById("emptyState"),
    chatForm: document.getElementById("chatForm"),
    userInput: document.getElementById("userInput"),
    modeStatusHint: document.getElementById("modeStatusHint"),
    regenBtn: document.getElementById("regenBtn"),
    stopBtn: document.getElementById("stopBtn"),
    sendBtn: document.getElementById("sendBtn"),
    settingsDrawer: document.getElementById("settingsDrawer"),
    closeSettingsBtn: document.getElementById("closeSettingsBtn"),
    settingSystemPrompt: document.getElementById("settingSystemPrompt"),
    settingMaxTokens: document.getElementById("settingMaxTokens"),
    settingTemperature: document.getElementById("settingTemperature"),
    settingTopP: document.getElementById("settingTopP"),
    settingSeed: document.getElementById("settingSeed"),
    settingStop: document.getElementById("settingStop"),
    resetSettingsBtn: document.getElementById("resetSettingsBtn"),
    settingsSavedHint: document.getElementById("settingsSavedHint"),
    perfPanel: document.getElementById("perfPanel"),
    closePerfBtn: document.getElementById("closePerfBtn"),
    statRss: document.getElementById("statRss"),
    statMlxActive: document.getElementById("statMlxActive"),
    statMlxPeak: document.getElementById("statMlxPeak"),
    statSwap: document.getElementById("statSwap"),
    statCtxLen: document.getElementById("statCtxLen"),
    statServerBusy: document.getElementById("statServerBusy"),
    statHitRate: document.getElementById("statHitRate"),
    statDecodeReqs: document.getElementById("statDecodeReqs"),
    statDemandMisses: document.getElementById("statDemandMisses"),
    statDemandBytes: document.getElementById("statDemandBytes"),
    statPrefetchBytes: document.getElementById("statPrefetchBytes"),
    statLastTokens: document.getElementById("statLastTokens"),
    statLastPrefill: document.getElementById("statLastPrefill"),
    statLastDecode: document.getElementById("statLastDecode"),
    statLastTime: document.getElementById("statLastTime"),
    clearPerfHistoryBtn: document.getElementById("clearPerfHistoryBtn"),
    sparkDecodeSummary: document.getElementById("sparkDecodeSummary"),
    sparkDecodeSvg: document.getElementById("sparkDecodeSvg"),
    sparkPrefillSummary: document.getElementById("sparkPrefillSummary"),
    sparkPrefillSvg: document.getElementById("sparkPrefillSvg"),
    modalOverlay: document.getElementById("modalOverlay"),
    modalTitle: document.getElementById("modalTitle"),
    modalMessage: document.getElementById("modalMessage"),
    modalInput: document.getElementById("modalInput"),
    modalCancelBtn: document.getElementById("modalCancelBtn"),
    modalConfirmBtn: document.getElementById("modalConfirmBtn"),
  };

  // --- Safe HTML & Markdown Rendering ---
  function escapeHtml(raw) {
    return String(raw ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function sanitizeUrl(url) {
    const trimmed = String(url ?? "").trim();
    if (/^(https?:\/\/|\/|#)/i.test(trimmed)) {
      return trimmed.replace(/"/g, "%22");
    }
    return "#";
  }

  function formatInlineMarkdown(escapedText) {
    const inlineCodes = [];
    let text = escapedText.replace(/`([^`\n]+)`/g, (_, code) => {
      const idx = inlineCodes.length;
      inlineCodes.push(`<code class="inline-code">${code}</code>`);
      return `\x00INLINECODE_${idx}\x00`;
    });

    // Links: [label](url)
    text = text.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, label, rawUrl) => {
      const unescapedUrl = rawUrl
        .replace(/&amp;/g, "&")
        .replace(/&lt;/g, "<")
        .replace(/&gt;/g, ">")
        .replace(/&quot;/g, '"')
        .replace(/&#39;/g, "'");
      const safeHref = sanitizeUrl(unescapedUrl);
      return `<a href="${safeHref}" target="_blank" rel="noopener noreferrer">${label}</a>`;
    });

    // Bold: **text**
    text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

    // Italic: *text*
    text = text.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>");

    // Restore inline code tokens
    text = text.replace(/\x00INLINECODE_(\d+)\x00/g, (_, i) => inlineCodes[Number(i)] || "");
    return text;
  }

  function renderSafeMarkdown(rawText) {
    if (!rawText) return "";
    const escaped = escapeHtml(rawText);
    const codeBlocks = [];

    // Extract fenced code blocks ```lang ... ```
    const withoutFences = escaped.replace(/```([a-zA-Z0-9_-]*)\n?([\s\S]*?)```/g, (_, lang, code) => {
      const idx = codeBlocks.length;
      const safeLang = lang || "text";
      codeBlocks.push(
        `<div class="code-block-wrap">` +
          `<div class="code-block-header"><span>${safeLang}</span></div>` +
          `<pre><code>${code.replace(/\n$/, "")}</code></pre>` +
        `</div>`
      );
      return `\x00CODEBLOCK_${idx}\x00`;
    });

    const paragraphs = withoutFences.split(/\n{2,}/);
    const htmlParts = paragraphs.map((para) => {
      const trimmed = para.trim();
      if (!trimmed) return "";
      if (/^\x00CODEBLOCK_\d+\x00$/.test(trimmed)) {
        return trimmed;
      }
      const lines = para.split("\n").map((line) => formatInlineMarkdown(line));
      return `<p>${lines.join("<br>")}</p>`;
    });

    let finalHtml = htmlParts.join("");
    finalHtml = finalHtml.replace(/\x00CODEBLOCK_(\d+)\x00/g, (_, i) => codeBlocks[Number(i)] || "");
    return finalHtml;
  }

  // --- LocalStorage Helpers ---
  function loadState() {
    try {
      const savedTheme = localStorage.getItem(STORAGE_KEYS.THEME);
      if (savedTheme === "dark" || savedTheme === "light") {
        document.documentElement.setAttribute("data-theme", savedTheme);
      }
      const savedSettings = JSON.parse(localStorage.getItem(STORAGE_KEYS.SETTINGS) || "null");
      if (savedSettings && typeof savedSettings === "object") {
        settings = { ...DEFAULT_SETTINGS, ...savedSettings };
      }
      const savedMode = localStorage.getItem(STORAGE_KEYS.MODE_THINKING);
      if (savedMode !== null) {
        enableThinking = savedMode === "true";
      }
      const savedPerf = JSON.parse(localStorage.getItem(STORAGE_KEYS.PERF_HISTORY) || "[]");
      if (Array.isArray(savedPerf)) {
        perfHistory = savedPerf.slice(-MAX_SPARK_POINTS);
      }
      const savedConvs = JSON.parse(localStorage.getItem(STORAGE_KEYS.CONVERSATIONS) || "[]");
      if (Array.isArray(savedConvs) && savedConvs.length > 0) {
        conversations = savedConvs;
      } else {
        conversations = [createBlankConversation()];
      }
      const savedActive = localStorage.getItem(STORAGE_KEYS.ACTIVE_CONV_ID);
      if (savedActive && conversations.some((c) => c.id === savedActive)) {
        activeConvId = savedActive;
      } else {
        activeConvId = conversations[0].id;
      }
    } catch (_) {
      conversations = [createBlankConversation()];
      activeConvId = conversations[0].id;
    }
  }

  function saveConversations() {
    try {
      localStorage.setItem(STORAGE_KEYS.CONVERSATIONS, JSON.stringify(conversations));
      if (activeConvId) {
        localStorage.setItem(STORAGE_KEYS.ACTIVE_CONV_ID, activeConvId);
      }
    } catch (_) {}
  }

  function saveSettings() {
    try {
      localStorage.setItem(STORAGE_KEYS.SETTINGS, JSON.stringify(settings));
    } catch (_) {}
  }

  function savePerfHistory() {
    try {
      localStorage.setItem(STORAGE_KEYS.PERF_HISTORY, JSON.stringify(perfHistory.slice(-MAX_SPARK_POINTS)));
    } catch (_) {}
  }

  function createBlankConversation() {
    return {
      id: "conv_" + Date.now() + "_" + Math.random().toString(36).slice(2, 7),
      title: "新對話",
      createdAt: Date.now(),
      messages: [],
    };
  }

  function getActiveConversation() {
    let conv = conversations.find((c) => c.id === activeConvId);
    if (!conv) {
      conv = createBlankConversation();
      conversations.unshift(conv);
      activeConvId = conv.id;
      saveConversations();
    }
    return conv;
  }

  // --- UI Helpers & Modals ---
  function showInlineError(msg) {
    el.inlineErrorMsg.textContent = msg;
    el.inlineErrorBox.classList.remove("hidden");
  }

  function hideInlineError() {
    el.inlineErrorBox.classList.add("hidden");
    el.inlineErrorMsg.textContent = "";
  }

  function setOfflineState(isOffline, detail) {
    if (isOffline) {
      el.offlineText.textContent = detail ? `伺服器未連線 (${detail})` : "伺服器未連線";
      el.offlineBanner.classList.remove("hidden");
    } else {
      el.offlineBanner.classList.add("hidden");
    }
  }

  function setBusyUI(busy) {
    if (busy) {
      el.busyIndicator.classList.add("busy");
      el.busyIndicator.classList.remove("idle");
      el.busyText.textContent = "忙碌中";
    } else {
      el.busyIndicator.classList.remove("busy");
      el.busyIndicator.classList.add("idle");
      el.busyText.textContent = "閒置";
    }
  }

  function openModal({ title, message, inputValue = null, onConfirm }) {
    el.modalTitle.textContent = title;
    el.modalMessage.textContent = message;
    if (inputValue !== null) {
      el.modalInput.value = inputValue;
      el.modalInput.classList.remove("hidden");
    } else {
      el.modalInput.classList.add("hidden");
    }
    modalCallback = onConfirm;
    el.modalOverlay.classList.remove("hidden");
    if (inputValue !== null) {
      el.modalInput.focus();
      el.modalInput.select();
    } else {
      el.modalConfirmBtn.focus();
    }
  }

  function closeModal() {
    el.modalOverlay.classList.add("hidden");
    modalCallback = null;
  }

  // --- Mode & Settings Sync ---
  function updateModeUI() {
    if (enableThinking) {
      el.modeThinkBtn.classList.add("active");
      el.modeThinkBtn.setAttribute("aria-pressed", "true");
      el.modeChatBtn.classList.remove("active");
      el.modeChatBtn.setAttribute("aria-pressed", "false");
      el.modeStatusHint.textContent = "目前模式：思考模式 (enable_thinking: true)";
    } else {
      el.modeChatBtn.classList.add("active");
      el.modeChatBtn.setAttribute("aria-pressed", "true");
      el.modeThinkBtn.classList.remove("active");
      el.modeThinkBtn.setAttribute("aria-pressed", "false");
      el.modeStatusHint.textContent = "目前模式：一般對話 (enable_thinking: false)";
    }
    try {
      localStorage.setItem(STORAGE_KEYS.MODE_THINKING, String(enableThinking));
    } catch (_) {}
  }

  function syncSettingsToInputs() {
    el.settingSystemPrompt.value = settings.systemPrompt || "";
    el.settingMaxTokens.value = settings.maxTokens ?? 1024;
    el.settingTemperature.value = settings.temperature ?? 0.7;
    el.settingTopP.value = settings.topP ?? 0.9;
    el.settingSeed.value = settings.seed ?? "";
    el.settingStop.value = settings.stop ?? "";
  }

  function readSettingsFromInputs() {
    const maxTok = parseInt(el.settingMaxTokens.value, 10);
    const temp = parseFloat(el.settingTemperature.value);
    const topP = parseFloat(el.settingTopP.value);
    settings = {
      systemPrompt: el.settingSystemPrompt.value,
      maxTokens: Number.isFinite(maxTok) && maxTok > 0 ? maxTok : 1024,
      temperature: Number.isFinite(temp) ? temp : 0.7,
      topP: Number.isFinite(topP) ? topP : 0.9,
      seed: el.settingSeed.value.trim(),
      stop: el.settingStop.value,
    };
    saveSettings();
    el.settingsSavedHint.textContent = "已儲存 ✓";
  }

  // --- Conversation List & Messages Rendering ---
  function renderConversationList() {
    el.convList.innerHTML = "";
    conversations.forEach((conv) => {
      const li = document.createElement("li");
      li.className = "conv-item" + (conv.id === activeConvId ? " active" : "");
      li.setAttribute("role", "option");
      li.setAttribute("aria-selected", conv.id === activeConvId ? "true" : "false");

      const titleSpan = document.createElement("span");
      titleSpan.className = "conv-title";
      titleSpan.textContent = conv.title || "新對話";
      titleSpan.addEventListener("click", () => {
        if (isGenerating) return;
        activeConvId = conv.id;
        saveConversations();
        renderConversationList();
        renderMessages();
      });

      const actionsDiv = document.createElement("div");
      actionsDiv.className = "conv-actions";

      const renameBtn = document.createElement("button");
      renameBtn.type = "button";
      renameBtn.className = "conv-action-btn";
      renameBtn.title = "重新命名對話";
      renameBtn.setAttribute("aria-label", "重新命名對話");
      renameBtn.textContent = "✎";
      renameBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        openModal({
          title: "重新命名對話",
          message: "請輸入新的對話標題：",
          inputValue: conv.title || "",
          onConfirm: (val) => {
            const trimmed = (val || "").trim();
            if (trimmed) {
              conv.title = trimmed;
              saveConversations();
              renderConversationList();
            }
          },
        });
      });

      const delBtn = document.createElement("button");
      delBtn.type = "button";
      delBtn.className = "conv-action-btn";
      delBtn.title = "刪除對話";
      delBtn.setAttribute("aria-label", "刪除對話");
      delBtn.textContent = "🗑";
      delBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        if (isGenerating) return;
        openModal({
          title: "刪除對話",
          message: `確定要刪除「${conv.title}」嗎？此操作無法復原。`,
          onConfirm: () => {
            conversations = conversations.filter((c) => c.id !== conv.id);
            if (conversations.length === 0) {
              conversations.push(createBlankConversation());
            }
            if (activeConvId === conv.id) {
              activeConvId = conversations[0].id;
            }
            saveConversations();
            renderConversationList();
            renderMessages();
          },
        });
      });

      actionsDiv.appendChild(renameBtn);
      actionsDiv.appendChild(delBtn);
      li.appendChild(titleSpan);
      li.appendChild(actionsDiv);
      el.convList.appendChild(li);
    });
  }

  function formatMetricsHtml(m) {
    if (!m) return "";
    const pills = [];
    if (m.promptTokens != null) pills.push(`<span class="metric-pill">輸入: ${m.promptTokens} tok</span>`);
    if (m.completionTokens != null) pills.push(`<span class="metric-pill">輸出: ${m.completionTokens} tok</span>`);
    if (m.ttftMs != null) pills.push(`<span class="metric-pill">TTFT: ${m.ttftMs} ms</span>`);
    if (m.prefillTps != null) pills.push(`<span class="metric-pill">Prefill: ${Number(m.prefillTps).toFixed(1)} tok/s</span>`);
    if (m.decodeTps != null) pills.push(`<span class="metric-pill">Decode: ${Number(m.decodeTps).toFixed(2)} tok/s</span>`);
    if (m.totalTimeS != null) pills.push(`<span class="metric-pill">總耗時: ${Number(m.totalTimeS).toFixed(2)} s</span>`);
    return pills.length ? `<div class="message-metrics">${pills.join("")}</div>` : "";
  }

  function createMessageElement(msg, index, isLastAssistant) {
    const row = document.createElement("div");
    row.className = `message-row ${msg.role}`;
    row.dataset.index = String(index);

    const card = document.createElement("div");
    card.className = "message-card";

    // Header
    const header = document.createElement("div");
    header.className = "message-header";
    const roleLabel = document.createElement("span");
    roleLabel.className = "message-role";
    roleLabel.textContent = msg.role === "user" ? "使用者 (User)" : "助理 (Assistant)";

    const tools = document.createElement("div");
    tools.className = "message-tools";

    const copyBtn = document.createElement("button");
    copyBtn.type = "button";
    copyBtn.className = "btn btn-small";
    copyBtn.textContent = "複製";
    copyBtn.setAttribute("aria-label", "複製訊息內容");
    copyBtn.addEventListener("click", () => {
      const textToCopy = msg.content || msg.reasoning_content || "";
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(textToCopy).then(() => {
          copyBtn.textContent = "已複製 ✓";
          setTimeout(() => { copyBtn.textContent = "複製"; }, 1200);
        });
      }
    });
    tools.appendChild(copyBtn);

    if (isLastAssistant) {
      const regenInlineBtn = document.createElement("button");
      regenInlineBtn.type = "button";
      regenInlineBtn.className = "btn btn-small";
      regenInlineBtn.textContent = "↻ 重新生成";
      regenInlineBtn.setAttribute("aria-label", "重新生成此回覆");
      regenInlineBtn.addEventListener("click", () => regenerateLastAnswer());
      tools.appendChild(regenInlineBtn);
    }

    header.appendChild(roleLabel);
    header.appendChild(tools);
    card.appendChild(header);

    // Thinking block (reasoning_content)
    const thinkDetails = document.createElement("details");
    thinkDetails.className = "thinking-block" + (msg.reasoning_content ? "" : " hidden");
    if (msg._thinkingOpen) {
      thinkDetails.open = true;
    }
    const thinkSummary = document.createElement("summary");
    thinkSummary.textContent = "思考過程 (reasoning_content)";
    const thinkBody = document.createElement("div");
    thinkBody.className = "thinking-content";
    thinkBody.textContent = msg.reasoning_content || "";
    thinkDetails.appendChild(thinkSummary);
    thinkDetails.appendChild(thinkBody);
    card.appendChild(thinkDetails);

    // Tool calls block
    const toolBlock = document.createElement("div");
    toolBlock.className = "tool-calls-block" + (msg.tool_calls && msg.tool_calls.length ? "" : " hidden");
    const toolTitle = document.createElement("div");
    toolTitle.className = "tool-calls-title";
    toolTitle.textContent = "工具呼叫 (tool_calls)";
    const toolPre = document.createElement("pre");
    toolPre.className = "tool-calls-pre";
    toolPre.textContent = msg.tool_calls && msg.tool_calls.length ? JSON.stringify(msg.tool_calls, null, 2) : "";
    toolBlock.appendChild(toolTitle);
    toolBlock.appendChild(toolPre);
    card.appendChild(toolBlock);

    // Main content
    const contentDiv = document.createElement("div");
    contentDiv.className = "message-content";
    contentDiv.innerHTML = renderSafeMarkdown(msg.content || "");
    card.appendChild(contentDiv);

    // Metrics container
    const metricsWrap = document.createElement("div");
    metricsWrap.className = "metrics-wrap";
    if (msg.role === "assistant" && msg.metrics) {
      metricsWrap.innerHTML = formatMetricsHtml(msg.metrics);
    }
    card.appendChild(metricsWrap);

    row.appendChild(card);
    return {
      row,
      thinkDetails,
      thinkBody,
      toolBlock,
      toolPre,
      contentDiv,
      metricsWrap,
    };
  }

  function renderMessages() {
    const conv = getActiveConversation();
    el.messagesContainer.innerHTML = "";
    if (!conv.messages || conv.messages.length === 0) {
      el.emptyState.classList.remove("hidden");
      el.messagesContainer.appendChild(el.emptyState);
      return;
    }
    el.emptyState.classList.add("hidden");

    let lastAssistantIdx = -1;
    for (let i = conv.messages.length - 1; i >= 0; i--) {
      if (conv.messages[i].role === "assistant") {
        lastAssistantIdx = i;
        break;
      }
    }

    conv.messages.forEach((msg, idx) => {
      const dom = createMessageElement(msg, idx, idx === lastAssistantIdx);
      el.messagesContainer.appendChild(dom.row);
    });
    el.messagesContainer.scrollTop = el.messagesContainer.scrollHeight;
  }

  // --- API Calls ---
  async function loadModels() {
    try {
      const resp = await fetch("/v1/models");
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`);
      }
      const data = await resp.json();
      setOfflineState(false);
      const list = Array.isArray(data.data) ? data.data : [];
      const currentVal = el.modelSelect.value;
      el.modelSelect.innerHTML = "";
      if (list.length === 0) {
        const opt = document.createElement("option");
        opt.value = "default";
        opt.textContent = "default";
        el.modelSelect.appendChild(opt);
      } else {
        list.forEach((m) => {
          const opt = document.createElement("option");
          opt.value = m.id;
          opt.textContent = m.id;
          el.modelSelect.appendChild(opt);
        });
        if (currentVal && list.some((m) => m.id === currentVal)) {
          el.modelSelect.value = currentVal;
        }
      }
    } catch (err) {
      setOfflineState(true, err.message || "無法取得模型列表");
    }
  }

  function parseStopStrings(rawStop) {
    if (!rawStop || !rawStop.trim()) return undefined;
    const parts = rawStop
      .split(/[,|\n]/)
      .map((s) => s.trim())
      .filter(Boolean);
    return parts.length > 0 ? parts : undefined;
  }

  async function sendChatRequest() {
    if (isGenerating) return;
    hideInlineError();

    const conv = getActiveConversation();
    if (!conv.messages.length) return;

    const apiMessages = [];
    if (settings.systemPrompt && settings.systemPrompt.trim()) {
      apiMessages.push({
        role: "system",
        content: settings.systemPrompt.trim(),
      });
    }
    conv.messages.forEach((m) => {
      apiMessages.push({
        role: m.role,
        content: m.content || "",
      });
    });

    const assistantMsg = {
      role: "assistant",
      content: "",
      reasoning_content: "",
      tool_calls: [],
      _thinkingOpen: enableThinking,
      metrics: null,
    };
    conv.messages.push(assistantMsg);
    saveConversations();

    el.emptyState.classList.add("hidden");
    const dom = createMessageElement(assistantMsg, conv.messages.length - 1, true);
    el.messagesContainer.appendChild(dom.row);
    el.messagesContainer.scrollTop = el.messagesContainer.scrollHeight;

    isGenerating = true;
    setBusyUI(true);
    el.stopBtn.classList.remove("hidden");
    el.sendBtn.disabled = true;
    el.regenBtn.disabled = true;

    activeAbortController = new AbortController();
    const tStart = performance.now();
    let tFirstToken = null;
    let answerStarted = false;

    const payload = {
      model: el.modelSelect.value || "default",
      messages: apiMessages,
      stream: true,
      max_tokens: Number(settings.maxTokens) || 1024,
      temperature: Number(settings.temperature ?? 0.7),
      top_p: Number(settings.topP ?? 0.9),
      enable_thinking: Boolean(enableThinking),
      stream_options: { include_usage: true },
    };

    if (settings.seed !== "" && settings.seed !== null && settings.seed !== undefined) {
      const parsedSeed = parseInt(settings.seed, 10);
      if (Number.isFinite(parsedSeed)) {
        payload.seed = parsedSeed;
      }
    }
    const stops = parseStopStrings(settings.stop);
    if (stops) {
      payload.stop = stops;
    }

    let finalUsage = null;
    let finalTimings = null;

    try {
      const response = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: activeAbortController.signal,
      });

      setOfflineState(false);

      if (!response.ok) {
        let errMessage = `HTTP ${response.status} ${response.statusText}`;
        try {
          const errJson = await response.json();
          if (errJson && errJson.error) {
            errMessage = errJson.error.message || JSON.stringify(errJson.error);
          } else {
            errMessage = JSON.stringify(errJson);
          }
        } catch (_) {
          const txt = await response.text().catch(() => "");
          if (txt) errMessage = txt;
        }
        throw new Error(errMessage);
      }

      const contentType = response.headers.get("Content-Type") || "";
      if (!contentType.includes("text/event-stream")) {
        // Non-stream fallback
        const json = await response.json();
        tFirstToken = performance.now();
        const choice = (json.choices && json.choices[0]) || {};
        const msg = choice.message || {};
        assistantMsg.reasoning_content = msg.reasoning_content || "";
        assistantMsg.content = msg.content || "";
        if (Array.isArray(msg.tool_calls)) {
          assistantMsg.tool_calls = msg.tool_calls;
        }
        assistantMsg._thinkingOpen = false;
        finalUsage = json.usage || null;
        finalTimings = json.timings || null;
        updateStreamingDom(dom, assistantMsg, false);
      } else {
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const rawLine of lines) {
            const line = rawLine.trim();
            if (!line || !line.startsWith("data:")) continue;
            const dataStr = line.slice(5).trim();
            if (dataStr === "[DONE]") continue;

            let chunk;
            try {
              chunk = JSON.parse(dataStr);
            } catch (parseErr) {
              showInlineError(`SSE JSON 解析錯誤: ${parseErr.message}`);
              continue;
            }

            if (chunk.error) {
              throw new Error(chunk.error.message || JSON.stringify(chunk.error));
            }

            if (chunk.usage) {
              finalUsage = chunk.usage;
            }
            if (chunk.timings) {
              finalTimings = chunk.timings;
              if (chunk.timings.decode_tok_per_s != null) {
                el.liveDecodeTps.textContent = Number(chunk.timings.decode_tok_per_s).toFixed(2);
              }
            }

            const choice = Array.isArray(chunk.choices) && chunk.choices[0];
            if (!choice) continue;
            const delta = choice.delta || {};

            if (delta.reasoning_content) {
              if (tFirstToken === null) tFirstToken = performance.now();
              assistantMsg.reasoning_content += delta.reasoning_content;
              if (!answerStarted) {
                assistantMsg._thinkingOpen = true;
              }
            }

            if (Array.isArray(delta.tool_calls)) {
              if (tFirstToken === null) tFirstToken = performance.now();
              delta.tool_calls.forEach((tc) => {
                const idx = tc.index ?? 0;
                if (!assistantMsg.tool_calls[idx]) {
                  assistantMsg.tool_calls[idx] = {
                    id: tc.id || "",
                    type: tc.type || "function",
                    function: { name: "", arguments: "" },
                  };
                }
                if (tc.id) assistantMsg.tool_calls[idx].id = tc.id;
                if (tc.function) {
                  if (tc.function.name) assistantMsg.tool_calls[idx].function.name += tc.function.name;
                  if (tc.function.arguments) assistantMsg.tool_calls[idx].function.arguments += tc.function.arguments;
                }
              });
            }

            if (delta.content) {
              if (tFirstToken === null) tFirstToken = performance.now();
              if (!answerStarted) {
                answerStarted = true;
                // Collapse reasoning block when the answer starts
                assistantMsg._thinkingOpen = false;
              }
              assistantMsg.content += delta.content;
            }

            updateStreamingDom(dom, assistantMsg, !answerStarted && Boolean(assistantMsg.reasoning_content));
          }
        }
      }

      const tEnd = performance.now();
      const totalTimeS = (tEnd - tStart) / 1000;
      const ttftMs = tFirstToken !== null ? Math.round(tFirstToken - tStart) : Math.round(totalTimeS * 1000);
      const promptTokens = finalUsage ? finalUsage.prompt_tokens : null;
      const completionTokens = finalUsage ? finalUsage.completion_tokens : null;
      const prefillS = finalTimings ? finalTimings.prefill_s : null;
      const decodeTps = finalTimings && finalTimings.decode_tok_per_s != null
        ? finalTimings.decode_tok_per_s
        : (completionTokens && totalTimeS > 0 ? completionTokens / totalTimeS : null);
      const prefillTps = finalTimings && finalTimings.prefill_tok_per_s != null
        ? finalTimings.prefill_tok_per_s
        : (promptTokens && prefillS && prefillS > 0 ? promptTokens / prefillS : null);

      assistantMsg._thinkingOpen = false;
      assistantMsg.metrics = {
        promptTokens,
        completionTokens,
        ttftMs,
        prefillTps,
        decodeTps,
        totalTimeS,
      };

      if (decodeTps != null) {
        el.liveDecodeTps.textContent = Number(decodeTps).toFixed(2);
      }
      if (decodeTps != null || prefillTps != null) {
        recordSparklinePoint(prefillTps || 0, decodeTps || 0);
      }

      updateStreamingDom(dom, assistantMsg, false);
      dom.metricsWrap.innerHTML = formatMetricsHtml(assistantMsg.metrics);
      saveConversations();
      pollStatusOnce();
    } catch (err) {
      if (err.name === "AbortError") {
        assistantMsg.content += (assistantMsg.content ? "\n\n" : "") + "*[已由使用者停止生成]*";
        assistantMsg._thinkingOpen = false;
        updateStreamingDom(dom, assistantMsg, false);
        saveConversations();
      } else {
        if (err.name === "TypeError" || /Failed to fetch|NetworkError/i.test(err.message)) {
          setOfflineState(true, "無法連線至推論伺服器");
        }
        showInlineError(err.message || "未知錯誤");
        if (!assistantMsg.content && !assistantMsg.reasoning_content) {
          conv.messages.pop();
          dom.row.remove();
        }
        saveConversations();
      }
    } finally {
      isGenerating = false;
      activeAbortController = null;
      setBusyUI(false);
      el.stopBtn.classList.add("hidden");
      el.sendBtn.disabled = false;
      el.regenBtn.disabled = false;
    }
  }

  function updateStreamingDom(dom, msg, keepThinkingOpen) {
    if (msg.reasoning_content) {
      dom.thinkDetails.classList.remove("hidden");
      dom.thinkBody.textContent = msg.reasoning_content;
      dom.thinkDetails.open = Boolean(keepThinkingOpen);
    } else {
      dom.thinkDetails.classList.add("hidden");
    }

    if (msg.tool_calls && msg.tool_calls.length > 0) {
      dom.toolBlock.classList.remove("hidden");
      dom.toolPre.textContent = JSON.stringify(msg.tool_calls, null, 2);
    } else {
      dom.toolBlock.classList.add("hidden");
    }

    dom.contentDiv.innerHTML = renderSafeMarkdown(msg.content || "");
    el.messagesContainer.scrollTop = el.messagesContainer.scrollHeight;
  }

  function handleUserSubmit() {
    if (isGenerating) return;
    const text = el.userInput.value.trim();
    if (!text) return;

    const conv = getActiveConversation();
    if (conv.messages.length === 0 && (conv.title === "新對話" || !conv.title)) {
      conv.title = text.slice(0, 28) + (text.length > 28 ? "…" : "");
    }
    conv.messages.push({
      role: "user",
      content: text,
    });
    el.userInput.value = "";
    saveConversations();
    renderConversationList();
    renderMessages();
    sendChatRequest();
  }

  function regenerateLastAnswer() {
    if (isGenerating) return;
    const conv = getActiveConversation();
    if (!conv.messages.length) return;

    if (conv.messages[conv.messages.length - 1].role === "assistant") {
      conv.messages.pop();
    }
    if (!conv.messages.length || conv.messages[conv.messages.length - 1].role !== "user") {
      return;
    }
    saveConversations();
    renderMessages();
    sendChatRequest();
  }

  // --- Export & Clear ---
  function downloadFile(filename, content, mime) {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  function exportCurrentJson() {
    const conv = getActiveConversation();
    const safeName = (conv.title || "conversation").replace(/[^\w\u4e00-\u9fa5-]+/g, "_");
    downloadFile(`${safeName}.json`, JSON.stringify(conv, null, 2), "application/json;charset=utf-8");
  }

  function exportCurrentMarkdown() {
    const conv = getActiveConversation();
    const lines = [`# ${conv.title || "對話紀錄"}`, ""];
    conv.messages.forEach((m) => {
      const roleName = m.role === "user" ? "User" : "Assistant";
      lines.push(`## ${roleName}`);
      if (m.reasoning_content) {
        lines.push("<details><summary>思考過程</summary>\n\n" + m.reasoning_content + "\n\n</details>\n");
      }
      if (m.tool_calls && m.tool_calls.length) {
        lines.push("```json\n" + JSON.stringify(m.tool_calls, null, 2) + "\n```\n");
      }
      lines.push(m.content || "");
      lines.push("");
    });
    const safeName = (conv.title || "conversation").replace(/[^\w\u4e00-\u9fa5-]+/g, "_");
    downloadFile(`${safeName}.md`, lines.join("\n"), "text/markdown;charset=utf-8");
  }

  // --- Performance Panel & Sparklines ---
  function formatBytes(bytes) {
    if (bytes == null || !Number.isFinite(bytes)) return "—";
    if (bytes >= 1024 ** 3) return (bytes / 1024 ** 3).toFixed(2) + " GiB";
    if (bytes >= 1024 ** 2) return (bytes / 1024 ** 2).toFixed(1) + " MiB";
    if (bytes >= 1024) return (bytes / 1024).toFixed(1) + " KiB";
    return bytes + " B";
  }

  function recordSparklinePoint(prefillTps, decodeTps, finishedAt) {
    if (finishedAt && finishedAt === lastRecordedFinishedAt) {
      return;
    }
    if (finishedAt) {
      lastRecordedFinishedAt = finishedAt;
    }
    perfHistory.push({
      prefillTps: Number(prefillTps) || 0,
      decodeTps: Number(decodeTps) || 0,
      ts: Date.now(),
    });
    if (perfHistory.length > MAX_SPARK_POINTS) {
      perfHistory = perfHistory.slice(-MAX_SPARK_POINTS);
    }
    savePerfHistory();
    renderSparklines();
  }

  function drawSparklineSvg(svgEl, summaryEl, values, colorHex) {
    const width = 260;
    const height = 56;
    const padX = 6;
    const padY = 6;
    svgEl.innerHTML = "";

    const valid = values.filter((v) => Number.isFinite(v) && v > 0);
    if (valid.length === 0) {
      summaryEl.textContent = "尚無資料";
      const emptyTxt = document.createElementNS("http://www.w3.org/2000/svg", "text");
      emptyTxt.setAttribute("x", String(width / 2));
      emptyTxt.setAttribute("y", String(height / 2 + 4));
      emptyTxt.setAttribute("text-anchor", "middle");
      emptyTxt.setAttribute("fill", "#8b949e");
      emptyTxt.setAttribute("font-size", "11");
      emptyTxt.textContent = "尚無推論紀錄 (最多保留 30 筆)";
      svgEl.appendChild(emptyTxt);
      return;
    }

    const minVal = Math.min(...valid);
    const maxVal = Math.max(...valid);
    const latest = valid[valid.length - 1];
    summaryEl.textContent = `最新 ${latest.toFixed(1)} | 最高 ${maxVal.toFixed(1)}`;

    const range = maxVal - minVal > 1e-6 ? maxVal - minVal : Math.max(maxVal, 1);
    const plotW = width - padX * 2;
    const plotH = height - padY * 2;

    const pts = valid.map((v, idx) => {
      const x = valid.length === 1 ? width / 2 : padX + (idx / (valid.length - 1)) * plotW;
      const norm = valid.length === 1 ? 0.5 : (v - minVal) / range;
      const y = padY + plotH - norm * plotH;
      return [x, y];
    });

    if (pts.length > 1) {
      const areaPoints = [
        `${pts[0][0].toFixed(1)},${height - padY}`,
        ...pts.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`),
        `${pts[pts.length - 1][0].toFixed(1)},${height - padY}`,
      ].join(" ");
      const polygon = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
      polygon.setAttribute("points", areaPoints);
      polygon.setAttribute("fill", colorHex);
      polygon.setAttribute("fill-opacity", "0.15");
      svgEl.appendChild(polygon);

      const polyline = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
      polyline.setAttribute("fill", "none");
      polyline.setAttribute("stroke", colorHex);
      polyline.setAttribute("stroke-width", "2");
      polyline.setAttribute("points", pts.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" "));
      svgEl.appendChild(polyline);
    }

    pts.forEach(([x, y], idx) => {
      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("cx", x.toFixed(1));
      circle.setAttribute("cy", y.toFixed(1));
      circle.setAttribute("r", idx === pts.length - 1 ? "3" : "2");
      circle.setAttribute("fill", colorHex);
      svgEl.appendChild(circle);
    });
  }

  function renderSparklines() {
    const decodeVals = perfHistory.map((p) => p.decodeTps);
    const prefillVals = perfHistory.map((p) => p.prefillTps);
    drawSparklineSvg(el.sparkDecodeSvg, el.sparkDecodeSummary, decodeVals, "#2f81f7");
    drawSparklineSvg(el.sparkPrefillSvg, el.sparkPrefillSummary, prefillVals, "#3fb950");
  }

  async function pollStatusOnce() {
    try {
      const resp = await fetch("/api/status");
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const s = await resp.json();
      setOfflineState(false);

      const serverBusy = Boolean(s.busy);
      setBusyUI(isGenerating || serverBusy);
      el.statServerBusy.textContent = serverBusy ? "忙碌中 (Busy)" : "閒置 (Idle)";
      el.statCtxLen.textContent = s.context_length != null ? `${s.context_length} tok` : "—";

      const mem = s.memory || {};
      el.statRss.textContent = mem.rss_gib != null ? `${Number(mem.rss_gib).toFixed(2)} GiB` : "—";
      el.statMlxActive.textContent = mem.mlx_active_gib != null ? `${Number(mem.mlx_active_gib).toFixed(2)} GiB` : "—";
      el.statMlxPeak.textContent = mem.mlx_peak_gib != null ? `${Number(mem.mlx_peak_gib).toFixed(2)} GiB` : "—";
      el.statSwap.textContent = mem.swap_used_mb != null ? `${Number(mem.swap_used_mb).toFixed(1)} MB` : "—";

      const cache = s.expert_cache_since_start || {};
      el.statHitRate.textContent = cache.hit_rate != null ? `${(Number(cache.hit_rate) * 100).toFixed(2)}%` : "—";
      el.statDecodeReqs.textContent = cache.decode_requests != null ? String(cache.decode_requests) : "—";
      el.statDemandMisses.textContent = cache.demand_misses != null ? String(cache.demand_misses) : "—";
      el.statDemandBytes.textContent = formatBytes(cache.demand_read_bytes);
      el.statPrefetchBytes.textContent = formatBytes(cache.prefetch_read_bytes);

      const last = s.last_request || {};
      if (last && (last.prompt_tokens != null || last.completion_tokens != null)) {
        el.statLastTokens.textContent = `${last.prompt_tokens ?? 0} / ${last.completion_tokens ?? 0} tok`;
        const preTps = last.prefill_tok_per_s != null ? `${Number(last.prefill_tok_per_s).toFixed(1)} tok/s` : "—";
        const preSec = last.prefill_s != null ? ` (${Number(last.prefill_s).toFixed(2)}s)` : "";
        el.statLastPrefill.textContent = preTps + preSec;

        const decTps = last.decode_tok_per_s != null ? `${Number(last.decode_tok_per_s).toFixed(2)} tok/s` : "—";
        const decSec = last.decode_s != null ? ` (${Number(last.decode_s).toFixed(2)}s)` : "";
        el.statLastDecode.textContent = decTps + decSec;

        if (last.decode_tok_per_s != null && !isGenerating) {
          el.liveDecodeTps.textContent = Number(last.decode_tok_per_s).toFixed(2);
        }

        const reqSec = last.request_s != null ? `${Number(last.request_s).toFixed(2)}s` : "—";
        const reason = last.finish_reason ? ` [${last.finish_reason}]` : "";
        el.statLastTime.textContent = reqSec + reason;

        if (last.finished_at && perfHistory.length === 0) {
          recordSparklinePoint(last.prefill_tok_per_s || 0, last.decode_tok_per_s || 0, last.finished_at);
        }
      }
    } catch (err) {
      setOfflineState(true, "無法取得伺服器狀態");
    }
  }

  function updateStatusPolling() {
    if (statusPollTimer) {
      clearInterval(statusPollTimer);
      statusPollTimer = null;
    }
    const isPerfVisible = !el.perfPanel.classList.contains("hidden") && document.visibilityState !== "hidden";
    if (isPerfVisible) {
      pollStatusOnce();
      statusPollTimer = setInterval(pollStatusOnce, 2000);
    }
  }

  // --- Event Listeners ---
  function bindEvents() {
    el.retryConnBtn.addEventListener("click", () => {
      loadModels();
      pollStatusOnce();
    });

    el.toggleSidebarBtn.addEventListener("click", () => {
      el.convSidebar.classList.toggle("collapsed");
    });

    el.modeChatBtn.addEventListener("click", () => {
      enableThinking = false;
      updateModeUI();
    });

    el.modeThinkBtn.addEventListener("click", () => {
      enableThinking = true;
      updateModeUI();
    });

    el.toggleSettingsBtn.addEventListener("click", () => {
      const hidden = el.settingsDrawer.classList.toggle("hidden");
      el.toggleSettingsBtn.setAttribute("aria-expanded", hidden ? "false" : "true");
    });

    el.closeSettingsBtn.addEventListener("click", () => {
      el.settingsDrawer.classList.add("hidden");
      el.toggleSettingsBtn.setAttribute("aria-expanded", "false");
    });

    el.togglePerfBtn.addEventListener("click", () => {
      const hidden = el.perfPanel.classList.toggle("hidden");
      el.togglePerfBtn.setAttribute("aria-expanded", hidden ? "false" : "true");
      updateStatusPolling();
    });

    el.closePerfBtn.addEventListener("click", () => {
      el.perfPanel.classList.add("hidden");
      el.togglePerfBtn.setAttribute("aria-expanded", "false");
      updateStatusPolling();
    });

    document.addEventListener("visibilitychange", updateStatusPolling);

    el.themeToggleBtn.addEventListener("click", () => {
      const root = document.documentElement;
      const current = root.getAttribute("data-theme");
      const systemDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
      let next;
      if (current === "dark") next = "light";
      else if (current === "light") next = "dark";
      else next = systemDark ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try {
        localStorage.setItem(STORAGE_KEYS.THEME, next);
      } catch (_) {}
    });

    el.newConvBtn.addEventListener("click", () => {
      if (isGenerating) return;
      const conv = createBlankConversation();
      conversations.unshift(conv);
      activeConvId = conv.id;
      saveConversations();
      renderConversationList();
      renderMessages();
      el.userInput.focus();
    });

    el.exportJsonBtn.addEventListener("click", exportCurrentJson);
    el.exportMdBtn.addEventListener("click", exportCurrentMarkdown);

    el.clearChatBtn.addEventListener("click", () => {
      if (isGenerating) return;
      const conv = getActiveConversation();
      if (!conv.messages.length) return;
      openModal({
        title: "清空對話確認",
        message: "確定要清空目前對話的所有訊息嗎？",
        onConfirm: () => {
          conv.messages = [];
          conv.title = "新對話";
          saveConversations();
          renderConversationList();
          renderMessages();
        },
      });
    });

    el.dismissErrorBtn.addEventListener("click", hideInlineError);

    el.chatForm.addEventListener("submit", (e) => {
      e.preventDefault();
      handleUserSubmit();
    });

    el.userInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        handleUserSubmit();
      }
    });

    el.stopBtn.addEventListener("click", () => {
      if (activeAbortController) {
        activeAbortController.abort();
      }
    });

    el.regenBtn.addEventListener("click", regenerateLastAnswer);

    [
      el.settingSystemPrompt,
      el.settingMaxTokens,
      el.settingTemperature,
      el.settingTopP,
      el.settingSeed,
      el.settingStop,
    ].forEach((input) => {
      input.addEventListener("input", readSettingsFromInputs);
      input.addEventListener("change", readSettingsFromInputs);
    });

    el.resetSettingsBtn.addEventListener("click", () => {
      settings = { ...DEFAULT_SETTINGS };
      saveSettings();
      syncSettingsToInputs();
      el.settingsSavedHint.textContent = "已恢復預設值";
    });

    el.clearPerfHistoryBtn.addEventListener("click", () => {
      perfHistory = [];
      savePerfHistory();
      renderSparklines();
    });

    el.modalCancelBtn.addEventListener("click", closeModal);
    el.modalConfirmBtn.addEventListener("click", () => {
      const cb = modalCallback;
      const val = el.modalInput.value;
      closeModal();
      if (typeof cb === "function") {
        cb(val);
      }
    });
    el.modalInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        el.modalConfirmBtn.click();
      } else if (e.key === "Escape") {
        closeModal();
      }
    });
  }

  // --- Initialization ---
  function init() {
    loadState();
    updateModeUI();
    syncSettingsToInputs();
    renderConversationList();
    renderMessages();
    renderSparklines();
    bindEvents();
    loadModels();
    updateStatusPolling();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
