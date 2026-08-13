"use strict";

const runtimeConfig = Object.freeze(window.AUTO_NEXUS_CONFIG || {});
const cloudApiBase = String(runtimeConfig.apiBaseUrl || "").trim().replace(/\/$/, "");

const state = {
  source: "path",
  files: [],
  profile: null,
  runs: [],
  selectedRun: null,
  health: null,
  auth: { config: null, user: null, idToken: null, refreshToken: null, expiresAt: 0 },
  insights: null,
  insightsLoadingRun: null,
  monitoring: null,
  pipelineRotation: { x: -0.18, y: -0.28 },
  geometryRotation: { x: -0.22, y: 0.42 },
  geometryView: {
    zoom: 1,
    pointSize: 1.1,
    colorMode: "actual",
    showSurface: true,
    projectedPoints: [],
    hoveredRow: null,
    legendKey: "",
    hiddenCategories: new Set(),
  },
  evidenceUrls: [],
  computeTarget: "cloud",
  localAgent: {
    url: "http://127.0.0.1:8788",
    token: "",
    connected: false,
    capabilities: null,
  },
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function compactNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "--";
  return new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(number);
}

function formatScore(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(4) : "--";
}

function formatDuration(seconds) {
  const value = Number(seconds);
  if (!Number.isFinite(value)) return "--";
  if (value < 60) return `${value.toFixed(1)}s`;
  if (value < 3600) return `${Math.floor(value / 60)}m ${Math.round(value % 60)}s`;
  return `${Math.floor(value / 3600)}h ${Math.round((value % 3600) / 60)}m`;
}

function fileName(path) {
  return String(path || "Unknown dataset").split(/[\\/]/).filter(Boolean).pop() || "Unknown dataset";
}

function runScore(run) {
  const summary = run.summary || {};
  return summary.testing_accuracy ?? summary.held_out_testing_metric ?? summary.testing_r2 ?? null;
}

function llmDescriptor(run) {
  const config = run.config?.llm || {};
  if (config.mode === "offline") return "Offline report";
  if (config.mode === "environment") return "Server LLM";
  if (config.mode === "ollama") return `Ollama / ${config.model || "local"}`;
  if (config.mode === "byok") return `${config.provider || "BYOK"} / ${config.model || "model"}`;
  return "Reporting auto";
}

function toast(message, tone = "info") {
  const node = document.createElement("div");
  node.className = `toast ${tone}`;
  node.textContent = message;
  $("#toast-region").append(node);
  window.setTimeout(() => node.remove(), 4500);
}

function apiUrl(path, target = state.computeTarget) {
  if (/^https?:\/\//i.test(path)) return path;
  const base = target === "local_agent" ? state.localAgent.url.replace(/\/$/, "") : cloudApiBase;
  return `${base}${path.startsWith("/") ? path : `/${path}`}`;
}

async function api(url, options = {}, target = state.computeTarget) {
  if (target === "cloud") await refreshFirebaseToken();
  const headers = new Headers(options.headers || {});
  const token = target === "local_agent" ? state.localAgent.token : state.auth.idToken;
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const response = await fetch(apiUrl(url, target), { ...options, headers });
  let payload = null;
  try {
    payload = await response.json();
  } catch (_) {
    payload = {};
  }
  if (response.status === 401 && target === "cloud" && state.auth.config?.required) showAuthGate("Your session expired. Sign in again.");
  if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
  return payload;
}

function showAuthGate(message = "") {
  $("#auth-gate").hidden = false;
  if (message) {
    $("#auth-message").textContent = message;
    $("#auth-message").classList.add("error");
  }
}

function hideAuthGate() {
  $("#auth-gate").hidden = true;
  $("#auth-message").classList.remove("error");
}

function persistFirebaseSession(payload) {
  state.auth.idToken = payload.idToken || payload.id_token;
  state.auth.refreshToken = payload.refreshToken || payload.refresh_token;
  state.auth.expiresAt = Date.now() + Number(payload.expiresIn || payload.expires_in || 3600) * 1000;
  sessionStorage.setItem("autonexus.firebase", JSON.stringify({
    idToken: state.auth.idToken,
    refreshToken: state.auth.refreshToken,
    expiresAt: state.auth.expiresAt,
  }));
}

async function refreshFirebaseToken() {
  if (!state.auth.config?.required || !state.auth.refreshToken || Date.now() < state.auth.expiresAt - 60000) return;
  const key = state.auth.config.firebase.apiKey;
  const body = new URLSearchParams({ grant_type: "refresh_token", refresh_token: state.auth.refreshToken });
  const response = await fetch(`https://securetoken.googleapis.com/v1/token?key=${encodeURIComponent(key)}`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body,
  });
  if (!response.ok) {
    signOut();
    throw new Error("Firebase session refresh failed.");
  }
  persistFirebaseSession(await response.json());
}

async function signIn(event) {
  event.preventDefault();
  const button = $("#auth-form button");
  button.disabled = true;
  try {
    const key = state.auth.config.firebase.apiKey;
    const response = await fetch(`https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key=${encodeURIComponent(key)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email: $("#auth-email").value.trim(), password: $("#auth-password").value, returnSecureToken: true }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error?.message || "Firebase sign-in failed.");
    persistFirebaseSession(payload);
    state.auth.user = await api("/api/auth/me", {}, "cloud");
    renderIdentity();
    hideAuthGate();
    $("#auth-password").value = "";
    await refreshRuns();
  } catch (error) {
    showAuthGate(error.message);
  } finally {
    button.disabled = false;
  }
}

function signOut() {
  sessionStorage.removeItem("autonexus.firebase");
  state.auth.user = null;
  state.auth.idToken = null;
  state.auth.refreshToken = null;
  state.runs = [];
  renderIdentity();
  renderOverview();
  renderRuns();
  showAuthGate("Signed out. Authenticate to access your missions.");
}

function renderIdentity() {
  const chip = $("#identity-chip");
  if (!state.auth.user) {
    chip.hidden = true;
    return;
  }
  chip.hidden = state.auth.config?.mode === "local";
  $("#identity-label").textContent = state.auth.user.email || state.auth.user.name || state.auth.user.uid;
  $("#auth-mode-label").textContent = state.auth.config?.required ? "FIREBASE IDENTITY" : "LOCAL CONTROL PLANE";
  $("#auth-scope-label").textContent = state.auth.config?.required ? "OWNER ISOLATED" : "127.0.0.1";
}

function configureDatasetSources() {
  const pathButton = $('[data-source="path"]');
  const pathPane = $('[data-source-pane="path"]');
  const allowed = state.computeTarget === "local_agent" || state.auth.config?.local_paths_allowed !== false;
  pathButton.disabled = !allowed;
  pathPane.querySelector("small").textContent = allowed
    ? "Use a CSV/Excel file or an image folder available to this machine."
    : "Cloud workers cannot read paths on your computer; use browser upload or pair the local agent.";
  if (!allowed && state.source === "path") {
    state.source = "upload";
    $$('.source-option').forEach((item) => item.classList.toggle("active", item.dataset.source === "upload"));
    $$('[data-source-pane]').forEach((pane) => pane.classList.toggle("active", pane.dataset.sourcePane === "upload"));
  }
}

function navigate(view, smooth = true) {
  $$("[data-view-panel]").forEach((panel) => panel.classList.toggle("active", panel.dataset.viewPanel === view));
  $$(".rail-link").forEach((link) => link.classList.toggle("active", link.dataset.view === view));
  history.replaceState(null, "", `#${view}`);
  window.scrollTo({ top: 0, behavior: smooth ? "smooth" : "auto" });
  activateWorkbench(view);
}

function setDatasetProfile(profile) {
  state.profile = profile;
  $("#dataset-dossier").classList.remove("dormant");
  $("#dataset-status").textContent = "DATASET READY";
  $("#dataset-status").classList.add("ready");
  $("#launch-state").textContent = "Dataset verified / mission ready";
  $("#dossier-modality").textContent = String(profile.modality || "--").toUpperCase();
  if (profile.modality === "vision") {
    $("#dossier-scale").textContent = `${compactNumber(profile.images)} images`;
    $("#dossier-shape").textContent = `${compactNumber(profile.classes)} classes`;
    $("#target-column").value = "";
    $("#target-column").disabled = true;
    $("#task").value = "auto";
    $("#dossier-quality").textContent = "STRUCTURE OK";
  } else {
    $("#dossier-scale").textContent = profile.size_bytes ? `${(profile.size_bytes / 1048576).toFixed(1)} MiB` : "Uploaded";
    $("#dossier-shape").textContent = `${compactNumber((profile.columns || []).length)} columns`;
    $("#target-column").disabled = false;
    $("#dossier-quality").textContent = profile.missing_cells_in_sample ? `${compactNumber(profile.missing_cells_in_sample)} missing` : "SAMPLE CLEAN";
    const options = $("#target-options");
    options.replaceChildren(...(profile.columns || []).map((column) => {
      const option = document.createElement("option");
      option.value = column;
      return option;
    }));
    if (!$("#target-column").value && profile.target_candidates?.length) {
      $("#target-column").value = profile.target_candidates[0];
    }
  }
}

function resetDatasetProfile() {
  state.profile = null;
  $("#dataset-dossier").classList.add("dormant");
  $("#dataset-status").textContent = "NOT INSPECTED";
  $("#dataset-status").classList.remove("ready");
  $("#launch-state").textContent = "Awaiting dataset";
}

function parseCsvHeader(text) {
  const firstLine = text.split(/\r?\n/, 1)[0] || "";
  const values = [];
  let value = "";
  let quoted = false;
  for (let index = 0; index < firstLine.length; index += 1) {
    const character = firstLine[index];
    if (character === '"' && firstLine[index + 1] === '"' && quoted) {
      value += '"';
      index += 1;
    } else if (character === '"') {
      quoted = !quoted;
    } else if (character === "," && !quoted) {
      values.push(value.trim());
      value = "";
    } else {
      value += character;
    }
  }
  values.push(value.trim());
  return values.filter(Boolean);
}

async function profileFiles(files) {
  if (!files.length) return;
  state.files = files;
  const totalBytes = files.reduce((sum, file) => sum + file.size, 0);
  $("#upload-summary").textContent = `${files.length.toLocaleString()} file${files.length === 1 ? "" : "s"} / ${(totalBytes / 1048576).toFixed(1)} MiB selected`;
  if (files.length === 1 && /\.(csv|xlsx|xls)$/i.test(files[0].name)) {
    let columns = [];
    if (/\.csv$/i.test(files[0].name)) {
      columns = parseCsvHeader(await files[0].slice(0, 262144).text());
    }
    setDatasetProfile({
      modality: "tabular",
      size_bytes: files[0].size,
      columns,
      target_candidates: columns.slice(-8).reverse(),
      missing_cells_in_sample: null,
    });
    return;
  }
  const imageFiles = files.filter((file) => /\.(jpg|jpeg|png|bmp|gif|tif|tiff|webp)$/i.test(file.name));
  if (!imageFiles.length) {
    resetDatasetProfile();
    toast("No supported dataset files were selected.", "error");
    return;
  }
  const classes = new Set();
  imageFiles.forEach((file) => {
    const parts = (file.webkitRelativePath || file.name).split("/");
    if (parts.length > 2) classes.add(parts[1]);
  });
  setDatasetProfile({ modality: "vision", images: imageFiles.length, classes: classes.size || "auto" });
}

function statusLabel(status) {
  return String(status || "unknown").replaceAll("_", " ").toUpperCase();
}

function renderOverview() {
  const completed = state.runs.filter((run) => run.status === "completed");
  const active = state.runs.filter((run) => ["queued", "running"].includes(run.status));
  const scores = completed.map(runScore).map(Number).filter(Number.isFinite);
  $("#metric-runs").textContent = state.runs.length;
  $("#metric-active").textContent = active.length;
  $("#metric-best").textContent = scores.length ? Math.max(...scores).toFixed(4) : "--";
  const target = $("#recent-runs");
  if (!state.runs.length) {
    target.className = "compact-runs empty-state";
    target.innerHTML = '<span class="empty-glyph">+</span><p>No missions recorded yet.</p>';
    return;
  }
  target.className = "compact-runs";
  target.innerHTML = state.runs.slice(0, 4).map((run) => `
    <div class="compact-run" data-run-id="${escapeHtml(run.id)}" tabindex="0">
      <i class="${escapeHtml(run.status)}"></i>
      <div><strong>${escapeHtml(run.best_model || statusLabel(run.status))}</strong><small>${escapeHtml(fileName(run.dataset))}</small></div>
      <time>${escapeHtml(new Date(run.created_at).toLocaleDateString())}</time>
    </div>`).join("");
}

function renderRunCard(run) {
  const summary = run.summary || {};
  const score = runScore(run);
  return `
    <article class="run-card" data-run-id="${escapeHtml(run.id)}" tabindex="0">
      <div class="run-card-head"><span class="run-id">${escapeHtml(run.id)}</span><span class="run-status ${escapeHtml(run.status)}">${escapeHtml(statusLabel(run.status))}</span></div>
      <h3>${escapeHtml(run.best_model || run.config?.preset || "Auto search")}</h3>
      <div class="dataset-name">${escapeHtml(fileName(run.dataset))}</div>
      <div class="run-score-row">
        <div><span>TEST SCORE</span><strong>${formatScore(score)}</strong></div>
        <div><span>PIPELINE TIME</span><strong>${escapeHtml(formatDuration(summary.total_pipeline_seconds))}</strong></div>
      </div>
      <div class="progress-track"><i style="width:${Math.max(0, Math.min(100, Number(run.progress) || 0))}%"></i></div>
      <div class="run-message">${escapeHtml(run.message || "")}</div>
    </article>`;
}

function renderRuns() {
  const query = $("#run-search").value.trim().toLowerCase();
  const filtered = state.runs.filter((run) => [run.id, run.best_model, run.dataset, run.status].join(" ").toLowerCase().includes(query));
  $("#archive-total").textContent = state.runs.length;
  $("#archive-complete").textContent = state.runs.filter((run) => run.status === "completed").length;
  $("#archive-active").textContent = state.runs.filter((run) => ["queued", "running"].includes(run.status)).length;
  $("#run-grid").innerHTML = filtered.length
    ? filtered.map(renderRunCard).join("")
    : '<div class="empty-state panel"><span class="empty-glyph">⌕</span><p>No matching missions.</p></div>';
  populateRunSelectors();
}

function populateRunSelectors() {
  const completed = state.runs.filter((run) => run.status === "completed");
  ["pipeline-run", "explain-run", "monitor-run"].forEach((id) => {
    const select = $(`#${id}`);
    const previous = select.value;
    select.innerHTML = completed.length
      ? completed.map((run) => `<option value="${escapeHtml(run.id)}">${escapeHtml(run.best_model)} / ${escapeHtml(fileName(run.dataset))} / ${escapeHtml(run.id)}</option>`).join("")
      : '<option value="">No completed missions</option>';
    if (completed.some((run) => run.id === previous)) select.value = previous;
  });
}

function metric(summary, ...keys) {
  for (const key of keys) if (summary[key] !== undefined && summary[key] !== null) return summary[key];
  return null;
}

function renderRunDetail(run) {
  const summary = run.summary || {};
  const artifacts = run.artifacts || [];
  const events = run.events || [];
  const artifactLabels = {
    manifest: "Run manifest", model: "Model bundle", notebook: "Notebook only",
    analytics_bundle: "Runnable analytics bundle",
    explanation: "Explanation", html_report: "HTML report", search_profile: "Search profile",
    framework: "Framework metadata", drift_baseline: "Drift baseline",
  };
  $("#run-detail").innerHTML = `
    <div class="dialog-title">
      <span class="run-status ${escapeHtml(run.status)}">${escapeHtml(statusLabel(run.status))}</span>
      <h2>${escapeHtml(run.best_model || "Mission in progress")}</h2>
      <p>${escapeHtml(run.id)} / ${escapeHtml(fileName(run.dataset))} / ${escapeHtml(llmDescriptor(run))}</p>
    </div>
    <div class="dialog-metrics">
      <div><span>TRAIN</span><strong>${formatScore(metric(summary, "training_accuracy", "training_r2"))}</strong></div>
      <div><span>VALIDATION</span><strong>${formatScore(metric(summary, "validation_accuracy", "validation_r2"))}</strong></div>
      <div><span>HELD-OUT TEST</span><strong>${formatScore(runScore(run))}</strong></div>
      <div><span>RUNTIME</span><strong>${escapeHtml(formatDuration(summary.total_pipeline_seconds))}</strong></div>
    </div>
    ${run.error ? `<div class="error-box">${escapeHtml(run.error)}</div>` : ""}
    ${run.status === "completed" ? `<div class="deployment-bar">
      <p><strong>${run.deployment?.status === "active" ? "INFERENCE ENDPOINT ACTIVE" : "ONE-CLICK LOCAL DEPLOYMENT"}</strong>${run.deployment?.status === "active" ? escapeHtml(run.deployment.predict_url) : "Loads the sealed model into an authenticated endpoint in this Studio process."}</p>
      <div class="deployment-actions">
        <button class="mini-button" type="button" data-deploy-run="${escapeHtml(run.id)}" ${run.deployment?.status === "active" ? "disabled" : ""}>DEPLOY</button>
        <button class="mini-button muted" type="button" data-undeploy-run="${escapeHtml(run.id)}" ${run.deployment?.status === "active" ? "" : "disabled"}>STOP</button>
      </div>
    </div>` : ""}
    <div class="detail-section"><span>MISSION TELEMETRY</span>
      <div class="event-list">${events.map((event) => `
        <div class="event-row"><time>${escapeHtml(new Date(event.time).toLocaleString())}</time><b>${escapeHtml(event.name)}</b><span>${escapeHtml(event.message)}</span></div>`).join("") || '<div class="event-row"><span>No events recorded.</span></div>'}</div>
    </div>
    <div class="detail-section"><span>EVIDENCE BUNDLE</span>
      <div class="artifact-grid">${artifacts.map((name) => `
        <button class="artifact-link" type="button" data-artifact-run="${escapeHtml(run.id)}" data-artifact-name="${escapeHtml(name)}">
          <span>${escapeHtml(artifactLabels[name] || name)}</span><b>+</b>
        </button>`).join("") || '<span class="run-message">Artifacts appear after a successful run.</span>'}</div>
    </div>`;
}

async function openRun(runId) {
  try {
    const run = await api(`/api/runs/${encodeURIComponent(runId)}`);
    state.selectedRun = run.id;
    renderRunDetail(run);
    $("#run-dialog").showModal();
  } catch (error) {
    toast(error.message, "error");
  }
}

async function refreshRuns() {
  try {
    const payload = await api("/api/runs");
    state.runs = payload.runs || [];
    renderOverview();
    renderRuns();
    const activeView = $(".view.active")?.dataset.viewPanel;
    if (["pipeline", "explain"].includes(activeView)) {
      const selected = $(`#${activeView === "pipeline" ? "pipeline" : "explain"}-run`).value;
      if (selected && state.insights?.run_id !== selected) loadInsights(activeView);
    }
    if (activeView === "monitor" && $("#monitor-run").value && !state.monitoring) loadMonitoring();
    if (state.selectedRun && $("#run-dialog").open) {
      const selected = state.runs.find((run) => run.id === state.selectedRun);
      if (selected) renderRunDetail(selected);
    }
  } catch (error) {
    toast(`Run archive unavailable: ${error.message}`, "error");
  }
}

async function inspectPath() {
  const path = $("#dataset-path").value.trim();
  if (!path) return toast("Enter a dataset path first.", "error");
  const button = $("#inspect-button");
  button.textContent = "SCANNING";
  button.disabled = true;
  try {
    setDatasetProfile(await api("/api/datasets/inspect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    }));
    toast("Dataset recognized. Mission controls are ready.");
  } catch (error) {
    resetDatasetProfile();
    toast(error.message, "error");
  } finally {
    button.textContent = "INSPECT";
    button.disabled = false;
  }
}

function updateComputeControls() {
  const local = state.computeTarget === "local_agent";
  $("#local-agent-panel").hidden = !local;
  $$(".compute-option").forEach((option) => {
    option.classList.toggle("selected", option.querySelector("input").checked);
  });
  $("#compute-status").textContent = local
    ? (state.localAgent.connected ? "LOCAL AGENT PAIRED" : "PAIRING REQUIRED")
    : "RAILWAY CLOUD";
  configureDatasetSources();
  resetDatasetProfile();
}

async function connectLocalAgent() {
  const button = $("#local-agent-connect");
  const url = $("#local-agent-url").value.trim().replace(/\/$/, "");
  const token = $("#local-agent-token").value.trim();
  if (!/^http:\/\/(127\.0\.0\.1|localhost)(:\d+)?$/i.test(url)) {
    return toast("The local agent URL must use localhost or 127.0.0.1.", "error");
  }
  if (token.length < 20) return toast("Paste the full local-agent pairing token.", "error");
  state.localAgent.url = url;
  state.localAgent.token = token;
  state.localAgent.connected = false;
  button.disabled = true;
  button.textContent = "PAIRING";
  try {
    const capabilities = await api("/api/agent/capabilities", {}, "local_agent");
    state.localAgent.capabilities = capabilities;
    state.localAgent.connected = true;
    const gpu = capabilities.gpu?.available
      ? `${capabilities.gpu.name} / ${capabilities.gpu.backend}`
      : "CPU only; no supported GPU detected";
    $("#local-agent-note").textContent = `Paired. ${gpu}. Data and metadata stay on this machine.`;
    $("#local-agent-token").value = "";
    updateComputeControls();
    await refreshRuns();
    toast(`Local agent paired. ${gpu}.`);
  } catch (error) {
    state.localAgent.token = "";
    $("#local-agent-note").textContent = "Pairing failed. Check the agent, token, and allowed Vercel origin.";
    toast(error.message, "error");
  } finally {
    button.disabled = false;
    button.textContent = "PAIR AGENT";
  }
}

function missionConfig() {
  const llmMode = $("#llm-mode").value;
  const llmProvider = $("#llm-provider").value;
  return {
    target: $("#target-column").value.trim() || null,
    task: $("#task").value,
    preset: $("input[name='preset']:checked").value,
    test_size: Number($("#test-size").value) / 100,
    cv: Number($("#cv").value),
    models: $("#models").value.split(",").map((value) => value.trim()).filter(Boolean),
    backbones: $("#backbones").value.split(",").map((value) => value.trim()).filter(Boolean),
    max_time: $("#max-time").value.trim() || null,
    feature_engineering: $("#feature-engineering").checked,
    tune: $("#tune").checked,
    adapt_lora: $("#adapt-lora").checked,
    shap: $("#shap").checked,
    llm_config: {
      mode: llmMode,
      provider: llmMode === "byok" ? llmProvider : null,
      model: ["byok", "ollama"].includes(llmMode) ? $("#llm-model").value.trim() : null,
      api_key: llmMode === "byok" ? $("#llm-api-key").value.trim() : null,
      api_base: (
        llmMode === "ollama" || (llmMode === "byok" && llmProvider === "custom")
      ) ? $("#llm-api-base").value.trim() : null,
    },
    use_memory: $("#memory").checked,
    contribute_memory: $("#memory").checked,
    execution_target: state.computeTarget,
    local_gpu_consent: false,
  };
}

function updateLLMControls() {
  const mode = $("#llm-mode").value;
  const provider = $("#llm-provider").value;
  const hosted = mode === "byok";
  const ollama = mode === "ollama";
  const endpoint = ollama || (hosted && provider === "custom");
  $("#llm-provider-field").classList.toggle("is-hidden", !hosted);
  $("#llm-model-field").classList.toggle("is-hidden", !hosted && !ollama);
  $("#llm-key-field").classList.toggle("is-hidden", !hosted);
  $("#llm-endpoint-field").classList.toggle("is-hidden", !endpoint);
  $(".llm-console").classList.toggle("simple", !hosted && !ollama);
  const badge = $("#llm-security-badge");
  badge.className = "secret-badge";
  if (mode === "byok") {
    badge.classList.add("ephemeral");
    badge.innerHTML = "<i></i> KEY EPHEMERAL";
    $("#llm-privacy-title").textContent = "Memory-only credential boundary";
    $("#llm-privacy-copy").textContent = "Your key is sent only to this local server for this mission, never persisted, logged, or stored in the browser.";
    $("#llm-model").placeholder = "Enter the exact provider model ID";
  } else if (mode === "ollama") {
    badge.classList.add("ephemeral");
    badge.innerHTML = "<i></i> LOCAL MODEL";
    $("#llm-privacy-title").textContent = "Local inference endpoint";
    $("#llm-privacy-copy").textContent = "No hosted API key is required. The final run context is sent to your configured Ollama service.";
    $("#llm-model").placeholder = "Local Ollama model name";
    if (!$("#llm-api-base").value || $("#llm-api-base").value.includes("provider.example")) {
      $("#llm-api-base").value = "http://localhost:11434";
    }
  } else if (mode === "offline") {
    badge.classList.add("offline");
    badge.innerHTML = "<i></i> NO EXTERNAL CALL";
    $("#llm-privacy-title").textContent = "Deterministic offline explanation";
    $("#llm-privacy-copy").textContent = "Auto Nexus writes the Markdown report locally without contacting an LLM service.";
  } else {
    badge.innerHTML = "<i></i> SERVER ENV";
    $("#llm-privacy-title").textContent = "Server-managed credentials";
    $("#llm-privacy-copy").textContent = "Uses LLM_MODEL and the provider key already configured in the server environment.";
  }
}

async function submitMission(event) {
  event.preventDefault();
  const button = $("#launch-button");
  const config = missionConfig();
  if (state.computeTarget === "local_agent") {
    if (!state.localAgent.connected || !state.localAgent.token) {
      return toast("Pair the local agent before starting local training.", "error");
    }
    const gpu = state.localAgent.capabilities?.gpu;
    const device = gpu?.available ? `${gpu.name} (${gpu.backend})` : "local CPU";
    const permitted = window.confirm(
      `Allow AutoNexus to train this mission on ${device}?\n\n` +
      "The selected dataset and generated artifacts stay in the local agent workspace. " +
      "Permission applies only to this run."
    );
    if (!permitted) return toast("Local compute permission was not granted.");
    config.local_gpu_consent = true;
  }
  const form = new FormData();
  form.append("config", JSON.stringify(config));
  if (state.source === "path") {
    const path = $("#dataset-path").value.trim();
    if (!path) return toast("Enter and inspect a dataset path.", "error");
    form.append("dataset_path", path);
  } else {
    if (!state.files.length) return toast("Select a dataset to upload.", "error");
    state.files.forEach((file) => form.append("files", file, file.webkitRelativePath || file.name));
  }
  button.disabled = true;
  button.querySelector("span").textContent = "Transmitting mission";
  $("#launch-state").textContent = "Validating configuration...";
  try {
    const run = await api("/api/runs", { method: "POST", body: form });
    $("#llm-api-key").value = "";
    toast(`Mission ${run.id} entered the training queue.`);
    await refreshRuns();
    navigate("runs");
    await openRun(run.id);
  } catch (error) {
    $("#launch-state").textContent = "Configuration rejected";
    toast(error.message, "error");
  } finally {
    button.disabled = false;
    button.querySelector("span").textContent = "Initialize training mission";
  }
}

async function authenticatedBlob(url) {
  if (state.computeTarget === "cloud") await refreshFirebaseToken();
  const headers = new Headers();
  const token = state.computeTarget === "local_agent" ? state.localAgent.token : state.auth.idToken;
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const response = await fetch(apiUrl(url), { headers });
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try { detail = (await response.json()).detail || detail; } catch (_) { /* binary response */ }
    throw new Error(detail);
  }
  return response.blob();
}

async function downloadArtifact(runId, name) {
  try {
    const blob = await authenticatedBlob(`/api/runs/${encodeURIComponent(runId)}/artifacts/${encodeURIComponent(name)}`);
    const url = URL.createObjectURL(blob);
    if (name === "html_report") {
      window.open(url, "_blank", "noopener");
    } else {
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = name === "analytics_bundle" ? "analysis_bundle.zip" : "";
      anchor.click();
    }
    window.setTimeout(() => URL.revokeObjectURL(url), 60000);
  } catch (error) {
    toast(error.message, "error");
  }
}

async function loadDocuments() {
  try {
    const payload = await fetch(apiUrl("/api/documents", "cloud")).then((response) => response.json());
    $$('[data-document]').forEach((card) => {
      const item = payload.documents?.[card.dataset.document];
      card.classList.toggle("available", Boolean(item?.available));
      card.classList.toggle("unavailable", !item?.available);
      card.querySelector("small").textContent = item?.available ? "PDF ONLINE" : "AWAITING UPLOAD";
      card.setAttribute("aria-disabled", item?.available ? "false" : "true");
      if (item?.available) card.href = apiUrl(item.url, "cloud");
    });
  } catch (_) {
    $$('[data-document]').forEach((card) => card.classList.add("unavailable"));
  }
}

function rotatePoint(point, rotation) {
  const cy = Math.cos(rotation.y); const sy = Math.sin(rotation.y);
  const cx = Math.cos(rotation.x); const sx = Math.sin(rotation.x);
  const x1 = point.x * cy - point.z * sy;
  const z1 = point.x * sy + point.z * cy;
  return { x: x1, y: point.y * cx - z1 * sx, z: point.y * sx + z1 * cx };
}

function canvasContext(canvas) {
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(canvas.clientWidth, 320);
  const height = Math.max(canvas.clientHeight, 260);
  if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
  }
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  return { context, width, height };
}

function project(point, rotation, width, height, scale = 1) {
  const rotated = rotatePoint(point, rotation);
  const perspective = 4.6 / Math.max(2.2, 4.6 + rotated.z);
  return {
    x: width / 2 + rotated.x * scale * perspective,
    y: height / 2 - rotated.y * scale * perspective,
    z: rotated.z,
    perspective,
  };
}

function drawPipeline() {
  const canvas = $("#pipeline-canvas");
  const nodes = state.insights?.lineage || [];
  const { context, width, height } = canvasContext(canvas);
  if (!nodes.length) {
    context.fillStyle = "#7896a3"; context.font = "12px Cascadia Code"; context.textAlign = "center";
    context.fillText("SELECT A COMPLETED MISSION", width / 2, height / 2);
    return;
  }
  const world = nodes.map((node, index) => ({
    ...node,
    x: (index - (nodes.length - 1) / 2) * 1.15,
    y: Math.sin(index * 1.15) * 0.38,
    z: Math.cos(index * 1.15) * 0.55,
  }));
  const scale = Math.min(width / Math.max(nodes.length * 1.25, 8), height / 4.2);
  const projected = world.map((node) => ({ ...node, ...project(node, state.pipelineRotation, width, height, scale) }));
  context.lineWidth = 1;
  for (let index = 0; index < projected.length - 1; index += 1) {
    const a = projected[index]; const b = projected[index + 1];
    context.strokeStyle = "rgba(66,232,224,.42)";
    context.beginPath(); context.moveTo(a.x, a.y); context.lineTo(b.x, b.y); context.stroke();
    const t = (Date.now() / 1300 + index * .17) % 1;
    context.fillStyle = "#42e8e0"; context.beginPath(); context.arc(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, 2.4, 0, Math.PI * 2); context.fill();
  }
  projected.sort((a, b) => a.z - b.z).forEach((node, index) => {
    const size = 30 * node.perspective;
    context.save(); context.translate(node.x, node.y); context.rotate(Math.PI / 4);
    context.fillStyle = "rgba(8,31,42,.96)"; context.strokeStyle = index === projected.length - 1 ? "#74f0a7" : "#42e8e0";
    context.shadowColor = context.strokeStyle; context.shadowBlur = 15; context.lineWidth = 1.2;
    context.fillRect(-size, -size, size * 2, size * 2); context.strokeRect(-size, -size, size * 2, size * 2); context.restore();
    context.shadowBlur = 0; context.textAlign = "center"; context.fillStyle = "#eaf8f7"; context.font = "600 11px Bahnschrift";
    context.fillText(node.label, node.x, node.y + size + 23);
    context.fillStyle = "#7896a3"; context.font = "9px Cascadia Code";
    context.fillText(node.detail.slice(0, 34), node.x, node.y + size + 39);
  });
}

const GEOMETRY_PALETTE = [
  "#42e8e0", "#ffca57", "#ff786f", "#74f0a7", "#73a7ff",
  "#f08cff", "#ff9c55", "#a5efff", "#d5ef67", "#d09cff",
  "#ff719f", "#70d7a7",
];

function geometryDomain(points, key) {
  return [...new Set(points.map((item) => String(item[key] ?? "Unknown")))].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
}

function categoricalGeometryColor(value, domain) {
  const index = Math.max(domain.indexOf(String(value ?? "Unknown")), 0);
  return GEOMETRY_PALETTE[index % GEOMETRY_PALETTE.length];
}

function confidenceGeometryColor(value) {
  const confidence = Math.max(0, Math.min(1, Number(value) || 0));
  return `hsl(${12 + confidence * 148} 78% 60%)`;
}

function geometryColorValue(item, mode) {
  if (mode === "predicted") return String(item.predicted ?? "Unknown");
  if (mode === "correctness") return item.correct === true ? "Correct" : item.correct === false ? "Incorrect" : "Unknown";
  if (mode === "split") return String(item.split ?? "Unknown");
  if (mode === "confidence") return Number(item.confidence ?? item.y ?? 0);
  return String(item.label ?? "Unknown");
}

function geometryPointColor(item, mode, domain) {
  const value = geometryColorValue(item, mode);
  return mode === "confidence" ? confidenceGeometryColor(value) : categoricalGeometryColor(value, domain);
}

function formatGeometryNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "--";
  const absolute = Math.abs(number);
  if ((absolute > 0 && absolute < .001) || absolute >= 1e6) return number.toExponential(3);
  return new Intl.NumberFormat("en", { maximumFractionDigits: 4 }).format(number);
}

function formatGeometryAxisValue(item, axis, index) {
  if (axis?.role === "model output" && /confidence/i.test(axis?.label || "")) {
    const confidence = Number(item.confidence);
    return Number.isFinite(confidence) ? `${(confidence * 100).toFixed(2)}%` : "--";
  }
  return formatGeometryNumber([item._rawX, item._rawY, item._rawZ][index]);
}

function renderGeometryLegend(points) {
  const mode = state.geometryView.colorMode;
  const legend = $("#geometry-legend");
  if (!points.length) { legend.innerHTML = "<span>NO GEOMETRY LOADED</span>"; return; }
  if (mode === "confidence") {
    const key = "confidence";
    if (state.geometryView.legendKey === key) return;
    state.geometryView.legendKey = key;
    legend.innerHTML = `<span><i style="color:${confidenceGeometryColor(.1)};background:${confidenceGeometryColor(.1)}"></i>LOW CONFIDENCE</span><span><i style="color:${confidenceGeometryColor(.5)};background:${confidenceGeometryColor(.5)}"></i>MEDIUM</span><span><i style="color:${confidenceGeometryColor(.95)};background:${confidenceGeometryColor(.95)}"></i>HIGH CONFIDENCE</span>`;
    return;
  }
  const keyName = mode === "actual" ? "label" : mode;
  const domain = mode === "correctness" ? ["Correct", "Incorrect", "Unknown"] : geometryDomain(points, keyName === "predicted" ? "predicted" : keyName === "split" ? "split" : "label");
  const key = `${mode}:${domain.join("|")}`;
  if (state.geometryView.legendKey === key) return;
  state.geometryView.legendKey = key;
  legend.innerHTML = `<span>CLICK TO FILTER</span>` + domain.map((value) => {
    const color = categoricalGeometryColor(value, domain);
    const filtered = state.geometryView.hiddenCategories.has(value) ? " filtered" : "";
    return `<button type="button" class="${filtered}" data-geometry-category="${escapeHtml(value)}" aria-pressed="${filtered ? "true" : "false"}"><i style="color:${color};background:${color}"></i>${escapeHtml(value)}</button>`;
  }).join("");
}

function drawGeometryAxisLabel(context, text, point, color, width, height) {
  const label = String(text || "axis");
  context.font = "9px Cascadia Code";
  const measured = context.measureText(label).width;
  const x = Math.max(8, Math.min(width - measured - 18, point.x + 8));
  const y = Math.max(18, Math.min(height - 8, point.y - 8));
  context.fillStyle = "rgba(4,18,27,.88)";
  context.fillRect(x - 5, y - 12, measured + 10, 18);
  context.fillStyle = color;
  context.textAlign = "left";
  context.fillText(label, x, y);
}

function drawGeometry() {
  const canvas = $("#geometry-canvas");
  const geometry = state.insights?.geometry;
  const points = geometry?.points || [];
  const { context, width, height } = canvasContext(canvas);
  if (!points.length) {
    context.fillStyle = "#7896a3"; context.font = "12px Cascadia Code"; context.textAlign = "center";
    context.fillText(geometry?.error || "NO SAVED GEOMETRY FOR THIS RUN", width / 2, height / 2);
    return;
  }
  const dimensions = ["x", "y", "z"].map((axis) => {
    const values = points.map((item) => Number(item[axis]) || 0);
    const minimum = Math.min(...values); const maximum = Math.max(...values);
    return { center: (minimum + maximum) / 2, half: Math.max((maximum - minimum) / 2, 1e-9) };
  });
  const normalize = (item) => ({
    x: (Number(item.x) - dimensions[0].center) / dimensions[0].half,
    y: (Number(item.y) - dimensions[1].center) / dimensions[1].half,
    z: (Number(item.z) - dimensions[2].center) / dimensions[2].half,
  });
  const normalized = points.map((item, index) => ({
    ...item,
    _geometryKey: String(item.row_id ?? index),
    _rawX: Number(item.x),
    _rawY: Number(item.y),
    _rawZ: Number(item.z),
    ...normalize(item),
  }));
  const scale = Math.min(width, height) * .34 * state.geometryView.zoom;
  const projected = normalized.map((item) => ({ ...item, ...project(item, state.geometryRotation, width, height, scale) })).sort((a, b) => a.z - b.z);
  const axes = geometry.response_surface?.axes || geometry.axes || [
    { key: "x", label: "component 1" }, { key: "y", label: "component 2" }, { key: "z", label: "component 3" },
  ];
  const axisColors = ["#42e8e0", "#ffca57", "#73a7ff"];
  const axisLines = [[{x:-1,y:0,z:0},{x:1,y:0,z:0}],[{x:0,y:-1,z:0},{x:0,y:1,z:0}],[{x:0,y:0,z:-1},{x:0,y:0,z:1}]];
  context.lineWidth = 1;
  axisLines.forEach(([a,b], index) => {
    const pa = project(a,state.geometryRotation,width,height,scale); const pb = project(b,state.geometryRotation,width,height,scale);
    context.strokeStyle = `${axisColors[index]}55`;
    context.beginPath(); context.moveTo(pa.x,pa.y); context.lineTo(pb.x,pb.y); context.stroke();
    drawGeometryAxisLabel(context, `${["X","Y","Z"][index]}  ${axes[index]?.label || "axis"}`, pb, axisColors[index], width, height);
  });
  if (geometry.kind === "exact_logistic_plane" && geometry.exact_plane) {
    const coefficients = geometry.exact_plane.coefficients.map((value, index) => Number(value) * dimensions[index].half);
    const intercept = Number(geometry.exact_plane.intercept) + geometry.exact_plane.coefficients.reduce((sum, value, index) => sum + Number(value) * dimensions[index].center, 0);
    const solve = coefficients.map(Math.abs).indexOf(Math.max(...coefficients.map(Math.abs)));
    const varying = [0, 1, 2].filter((index) => index !== solve);
    const corners = [[-1,-1],[1,-1],[1,1],[-1,1]].map(([first, second]) => {
      const values = [0,0,0]; values[varying[0]] = first; values[varying[1]] = second;
      values[solve] = (-intercept - coefficients[varying[0]] * first - coefficients[varying[1]] * second) / coefficients[solve];
      return { x: values[0], y: values[1], z: values[2] };
    });
    if (corners.every((corner) => Number.isFinite(corner.x + corner.y + corner.z))) {
      const projectedPlane = corners.map((corner) => project(corner, state.geometryRotation, width, height, scale));
      context.fillStyle = "rgba(66,232,224,.1)"; context.strokeStyle = "rgba(146,255,245,.7)";
      context.beginPath(); context.moveTo(projectedPlane[0].x,projectedPlane[0].y);
      projectedPlane.slice(1).forEach((corner) => context.lineTo(corner.x,corner.y)); context.closePath(); context.fill(); context.stroke();
    }
  }
  if (state.geometryView.showSurface && geometry.response_surface?.vertices?.length) {
    const surface = geometry.response_surface;
    const vertices = surface.vertices.map((vertex) => project(normalize(vertex), state.geometryRotation, width, height, scale));
    const predictionDomain = geometryDomain(surface.vertices, "predicted");
    context.lineWidth = .9;
    for (let row = 0; row < surface.rows; row += 1) {
      for (let column = 0; column < surface.columns - 1; column += 1) {
        const index = row * surface.columns + column;
        const first = vertices[index]; const second = vertices[index + 1];
        context.strokeStyle = `${categoricalGeometryColor(surface.vertices[index].predicted, predictionDomain)}66`;
        context.beginPath(); context.moveTo(first.x, first.y); context.lineTo(second.x, second.y); context.stroke();
      }
    }
    for (let column = 0; column < surface.columns; column += 1) {
      for (let row = 0; row < surface.rows - 1; row += 1) {
        const index = row * surface.columns + column;
        const first = vertices[index]; const second = vertices[index + surface.columns];
        context.strokeStyle = `${categoricalGeometryColor(surface.vertices[index].predicted, predictionDomain)}66`;
        context.beginPath(); context.moveTo(first.x, first.y); context.lineTo(second.x, second.y); context.stroke();
      }
    }
  }
  if (geometry.adaptation_movement?.length) {
    context.strokeStyle = "rgba(146,255,245,.30)";
    context.lineWidth = .75;
    geometry.adaptation_movement.forEach((movement) => {
      const start = normalize({ x: movement.from[0], y: movement.from[1], z: movement.from[2] });
      const end = normalize({ x: movement.to[0], y: movement.to[1], z: movement.to[2] });
      const projectedStart = project(start, state.geometryRotation, width, height, scale);
      const projectedEnd = project(end, state.geometryRotation, width, height, scale);
      context.beginPath();
      context.moveTo(projectedStart.x, projectedStart.y);
      context.lineTo(projectedEnd.x, projectedEnd.y);
      context.stroke();
    });
  }
  const mode = state.geometryView.colorMode;
  const domain = mode === "correctness" ? ["Correct", "Incorrect", "Unknown"] : geometryDomain(points, mode === "predicted" ? "predicted" : mode === "split" ? "split" : "label");
  const visibleProjected = [];
  projected.forEach((item) => {
    if (mode !== "confidence" && state.geometryView.hiddenCategories.has(String(geometryColorValue(item, mode)))) return;
    const radius = Math.max(2.2, 3.5 * item.perspective * state.geometryView.pointSize);
    context.globalAlpha = item.split === "test" ? .95 : .48;
    context.fillStyle = geometryPointColor(item, mode, domain); context.beginPath(); context.arc(item.x,item.y,radius,0,Math.PI*2); context.fill();
    if (item.correct === false) { context.strokeStyle = "#ff786f"; context.lineWidth = 1.4; context.stroke(); }
    if (item._geometryKey === state.geometryView.hoveredRow) { context.strokeStyle = "#eaf8f7"; context.lineWidth = 2; context.stroke(); }
    item.screenRadius = radius;
    visibleProjected.push(item);
  });
  context.globalAlpha = 1;
  state.geometryView.projectedPoints = visibleProjected;
  renderGeometryLegend(points);
}

function bindCanvasRotation(canvas, rotation, draw) {
  let dragging = false; let lastX = 0; let lastY = 0;
  canvas.addEventListener("pointerdown", (event) => { dragging = true; lastX = event.clientX; lastY = event.clientY; canvas.setPointerCapture(event.pointerId); });
  canvas.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    rotation.y += (event.clientX - lastX) * .008; rotation.x += (event.clientY - lastY) * .008;
    lastX = event.clientX; lastY = event.clientY; draw();
  });
  canvas.addEventListener("pointerup", () => { dragging = false; });
  canvas.addEventListener("pointercancel", () => { dragging = false; });
}

function geometryTooltipHtml(item) {
  const axes = state.insights?.geometry?.response_surface?.axes || state.insights?.geometry?.axes || [
    { label: "component 1" }, { label: "component 2" }, { label: "component 3" },
  ];
  const correctness = item.correct === true ? "Correct" : item.correct === false ? "Incorrect" : "Not available";
  return `<b>SAMPLE ${escapeHtml(item.row_id ?? item._geometryKey)}</b>
    <span>${escapeHtml(axes[0]?.label || "X")}<strong>${escapeHtml(formatGeometryAxisValue(item, axes[0], 0))}</strong></span>
    <span>${escapeHtml(axes[1]?.label || "Y")}<strong>${escapeHtml(formatGeometryAxisValue(item, axes[1], 1))}</strong></span>
    <span>${escapeHtml(axes[2]?.label || "Z")}<strong>${escapeHtml(formatGeometryAxisValue(item, axes[2], 2))}</strong></span>
    <span>Actual<strong>${escapeHtml(item.label ?? "Unknown")}</strong></span>
    <span>Predicted<strong>${escapeHtml(item.predicted ?? "Not available")}</strong></span>
    <span>Result<strong>${escapeHtml(correctness)}</strong></span>`;
}

function updateGeometryHover(event) {
  const canvas = $("#geometry-canvas");
  const tooltip = $("#geometry-tooltip");
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left; const y = event.clientY - rect.top;
  let nearest = null; let nearestDistance = Infinity;
  state.geometryView.projectedPoints.forEach((item) => {
    const distance = Math.hypot(item.x - x, item.y - y);
    if (distance < Math.max(10, item.screenRadius * 2.2) && distance < nearestDistance) {
      nearest = item; nearestDistance = distance;
    }
  });
  if (!nearest) {
    if (state.geometryView.hoveredRow !== null) { state.geometryView.hoveredRow = null; drawGeometry(); }
    tooltip.hidden = true;
    return;
  }
  const changed = state.geometryView.hoveredRow !== nearest._geometryKey;
  state.geometryView.hoveredRow = nearest._geometryKey;
  tooltip.innerHTML = geometryTooltipHtml(nearest);
  tooltip.hidden = false;
  const stage = $("#geometry-stage");
  const left = Math.min(stage.clientWidth - 255, Math.max(10, x + 18));
  const top = Math.min(stage.clientHeight - 185, Math.max(10, y + 18));
  tooltip.style.left = `${left}px`; tooltip.style.top = `${top}px`;
  if (changed) drawGeometry();
}

function resetGeometryView(mode = "isometric") {
  state.geometryRotation.x = mode === "top" ? -Math.PI / 2 : -0.22;
  state.geometryRotation.y = mode === "top" ? 0 : 0.42;
  state.geometryView.zoom = 1;
  state.geometryView.hoveredRow = null;
  $("#geometry-tooltip").hidden = true;
  drawGeometry();
}

async function loadInsights(source = "pipeline") {
  const runId = $(`#${source}-run`).value;
  if (!runId) { state.insights = null; drawPipeline(); drawGeometry(); return; }
  if (state.insightsLoadingRun === runId) return;
  state.insightsLoadingRun = runId;
  try {
    state.insights = await api(`/api/runs/${encodeURIComponent(runId)}/insights`);
    const run = state.runs.find((item) => item.id === runId);
    $("#pipeline-title").textContent = `${run?.best_model || "Model"} / ${fileName(run?.dataset)}`;
    $("#lineage-fallback").innerHTML = state.insights.lineage.map((node, index) => `<li><b>${String(index + 1).padStart(2,"0")}</b><strong>${escapeHtml(node.label)}</strong><small>${escapeHtml(node.detail)}</small></li>`).join("");
    $("#geometry-mode").textContent = state.insights.geometry.kind.replaceAll("_", " ").toUpperCase();
    $("#geometry-title").textContent = state.insights.geometry.title;
    $("#geometry-truth").textContent = state.insights.geometry.truth_label;
    const geometryPoints = state.insights.geometry.points || [];
    const surface = state.insights.geometry.response_surface;
    const classCount = new Set(geometryPoints.map((item) => String(item.label ?? "Unknown"))).size;
    const evaluated = geometryPoints.filter((item) => item.correct != null);
    const errors = evaluated.filter((item) => item.correct === false).length;
    $("#geometry-height-guide").textContent = surface?.score_label || "Projected component";
    $("#geometry-stats").innerHTML = [
      ["SAMPLES", compactNumber(geometryPoints.length)],
      ["CLASSES", compactNumber(classCount)],
      ["ERRORS", evaluated.length ? `${compactNumber(errors)} / ${compactNumber(evaluated.length)}` : "N/A"],
      ["AXES", surface ? `${surface.feature_x} + ${surface.feature_z}` : "PROJECTED"],
      ["SELECTION", surface?.selection_method || "representation projection"],
    ].map(([label, value]) => `<span>${escapeHtml(label)}<b>${escapeHtml(value)}</b></span>`).join("");
    $("#geometry-notes").innerHTML = (state.insights.geometry.notes || []).map((note) => `<p>${escapeHtml(note)}</p>`).join("");
    $("#insight-warnings").innerHTML = (state.insights.warnings || []).map((warning) => `<p>${escapeHtml(warning)}</p>`).join("");
    const evidence = state.insights.explainability.evidence;
    $("#evidence-status").innerHTML = Object.entries(evidence).map(([name, available]) => `<div><span>${escapeHtml(name.replaceAll("_", " ").toUpperCase())}</span><b class="${available ? "" : "missing"}">${available ? "ONLINE" : "MISSING"}</b></div>`).join("");
    $("#local-explanation-note").textContent = state.insights.explainability.local_explanations;
    $("#attention-note").textContent = `${state.insights.explainability.vision_attention} ${state.insights.explainability.adaptation_movement}`;
    drawPipeline(); drawGeometry();
    if (source === "explain") await loadEvidenceImages(runId, evidence);
  } catch (error) {
    toast(error.message, "error");
  } finally {
    if (state.insightsLoadingRun === runId) state.insightsLoadingRun = null;
  }
}

async function loadEvidenceImages(runId, evidence) {
  state.evidenceUrls.forEach((url) => URL.revokeObjectURL(url)); state.evidenceUrls = [];
  const map = {
    feature_importance: "feature-importance",
    shap_summary: "shap-summary",
    shap_importance: "shap-importance",
    shap_dependence: "shap-dependence",
    shap_waterfall: "shap-waterfall",
    shap_decision: "shap-decision",
  };
  await Promise.all(Object.entries(map).map(async ([name, id]) => {
    const image = $(`#${id}-image`); const empty = $(`#${id}-empty`);
    image.hidden = true; empty.hidden = false;
    if (!evidence[name]) return;
    try {
      const blob = await authenticatedBlob(`/api/runs/${encodeURIComponent(runId)}/evidence/${name}`);
      const url = URL.createObjectURL(blob); state.evidenceUrls.push(url); image.src = url; image.hidden = false; empty.hidden = true;
    } catch (error) { empty.textContent = error.message; }
  }));
}

async function loadMonitoring() {
  const runId = $("#monitor-run").value;
  if (!runId) return;
  try {
    state.monitoring = await api(`/api/runs/${encodeURIComponent(runId)}/monitoring`);
    const baseline = state.monitoring.baseline || {};
    const deployment = state.monitoring.deployment || {};
    const telemetry = state.monitoring.deployment_telemetry || {};
    $("#monitor-summary").innerHTML = `<span class="kicker">SEALED BASELINE</span><h2>Reference population</h2><div class="monitor-kpis">
      <div><span>TRAINING SAMPLES</span><strong>${compactNumber(baseline.sample_count)}</strong></div><div><span>FEATURES</span><strong>${compactNumber(baseline.feature_count)}</strong></div>
      <div><span>EXPECTED ${escapeHtml(baseline.metric_name || "METRIC")}</span><strong>${formatScore(baseline.expected_metric)}</strong></div><div><span>INCREMENTAL</span><strong>${state.monitoring.incremental_supported ? "NATIVE" : "RETRAIN"}</strong></div>
      <div><span>ENDPOINT</span><strong>${escapeHtml((deployment.status || "inactive").toUpperCase())}</strong></div><div><span>REQUESTS / ERRORS</span><strong>${compactNumber(telemetry.request_count || 0)} / ${compactNumber(telemetry.error_count || 0)}</strong></div>
      <div><span>MEAN LATENCY</span><strong>${telemetry.mean_latency_ms == null ? "--" : `${Number(telemetry.mean_latency_ms).toFixed(1)} ms`}</strong></div><div><span>MEAN CONFIDENCE</span><strong>${formatScore(telemetry.mean_confidence)}</strong></div></div>`;
    $("#monitor-events").innerHTML = (state.monitoring.events || []).map((event) => {
      const outcome = event.severity === "insufficient_data" ? "Insufficient data" : event.drifted ? "Drift detected" : "Stable";
      return `<div class="event-row"><time>${escapeHtml(new Date((event.timestamp || 0) * 1000).toLocaleString())}</time><b>${escapeHtml(event.severity || "signal")}</b><span>${outcome} / ${compactNumber(event.sample_count)} samples</span></div>`;
    }).join("") || '<div class="event-row"><span>No monitoring observations recorded.</span></div>';
  } catch (error) { toast(error.message, "error"); }
}

async function evaluateIncrementalUpdate() {
  const runId = $("#monitor-run").value;
  try {
    const records = JSON.parse($("#incremental-records").value);
    const result = await api(`/api/runs/${encodeURIComponent(runId)}/incremental-update`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ records }),
    });
    $("#incremental-result").textContent = JSON.stringify(result, null, 2);
    toast(result.promoted ? "Challenger promoted." : "Challenger retained but not promoted.");
    await loadMonitoring();
    await refreshRuns();
  } catch (error) {
    $("#incremental-result").textContent = error.message;
    toast(error.message, "error");
  }
}

async function evaluateWhatIf() {
  const runId = $("#explain-run").value;
  try {
    const record = JSON.parse($("#what-if-record").value);
    const result = await api(`/api/runs/${encodeURIComponent(runId)}/what-if`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ record }),
    });
    $("#what-if-result").textContent = JSON.stringify(result, null, 2);
  } catch (error) {
    $("#what-if-result").textContent = error.message;
    toast(error.message, "error");
  }
}

async function observeBatch() {
  const runId = $("#monitor-run").value;
  try {
    const records = JSON.parse($("#monitor-records").value);
    const result = await api(`/api/runs/${encodeURIComponent(runId)}/monitoring/observe`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ records }) });
    $("#monitor-result").textContent = JSON.stringify(result, null, 2);
    await loadMonitoring(); await refreshRuns();
  } catch (error) { $("#monitor-result").textContent = error.message; toast(error.message, "error"); }
}

async function loadAudit() {
  try {
    const payload = await api("/api/audit");
    $("#audit-table").innerHTML = (payload.events || []).map((event) => `<div class="audit-row"><time>${escapeHtml(new Date(event.time).toLocaleString())}</time><code>${escapeHtml(event.run_id)}</code><b>${escapeHtml(event.name)}</b><span>${escapeHtml(event.message)} / ${escapeHtml(event.dataset)}</span></div>`).join("") || '<div class="empty-state"><p>No lifecycle events for this owner.</p></div>';
  } catch (error) { toast(error.message, "error"); }
}

async function deployRun(runId, stop = false) {
  try {
    await api(`/api/runs/${encodeURIComponent(runId)}/deploy`, { method: stop ? "DELETE" : "POST" });
    toast(stop ? "Inference endpoint stopped." : "Authenticated inference endpoint is active.");
    await refreshRuns(); await openRun(runId);
  } catch (error) { toast(error.message, "error"); }
}

function activateWorkbench(view) {
  if (view === "pipeline") loadInsights("pipeline");
  if (view === "explain") loadInsights("explain");
  if (view === "monitor") loadMonitoring();
  if (view === "audit") loadAudit();
}

function bindEvents() {
  $("#auth-form").addEventListener("submit", signIn);
  $("#identity-chip").addEventListener("click", signOut);
  $$('[data-navigate]').forEach((button) => button.addEventListener("click", () => navigate(button.dataset.navigate)));
  $$(".rail-link").forEach((link) => link.addEventListener("click", (event) => { event.preventDefault(); navigate(link.dataset.view); }));
  $$(".source-option").forEach((button) => button.addEventListener("click", () => {
    state.source = button.dataset.source;
    $$(".source-option").forEach((item) => item.classList.toggle("active", item === button));
    $$("[data-source-pane]").forEach((pane) => pane.classList.toggle("active", pane.dataset.sourcePane === state.source));
    resetDatasetProfile();
  }));
  $("#inspect-button").addEventListener("click", inspectPath);
  $("#dataset-path").addEventListener("keydown", (event) => { if (event.key === "Enter") { event.preventDefault(); inspectPath(); } });
  $("#dataset-path").addEventListener("input", resetDatasetProfile);
  $$("input[name='compute-target']").forEach((input) => input.addEventListener("change", async () => {
    state.computeTarget = input.value;
    updateComputeControls();
    if (state.computeTarget === "cloud") await refreshRuns();
  }));
  $("#local-agent-connect").addEventListener("click", connectLocalAgent);
  $("#file-input").addEventListener("change", (event) => profileFiles([...event.target.files]));
  $("#folder-input").addEventListener("change", (event) => profileFiles([...event.target.files]));
  const dropZone = $("#drop-zone");
  ["dragenter", "dragover"].forEach((name) => dropZone.addEventListener(name, (event) => { event.preventDefault(); dropZone.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach((name) => dropZone.addEventListener(name, (event) => { event.preventDefault(); dropZone.classList.remove("dragging"); }));
  dropZone.addEventListener("drop", (event) => profileFiles([...event.dataTransfer.files]));
  $("#test-size").addEventListener("input", (event) => { $("#test-size-output").textContent = `${event.target.value}%`; });
  $$("input[name='preset']").forEach((input) => input.addEventListener("change", () => {
    $$(".preset-card").forEach((card) => card.classList.toggle("selected", card.contains(input) && input.checked));
    if (input.value === "fast") $("#cv").value = "2";
    if (input.value === "accurate") { $("#cv").value = "5"; $("#tune").checked = true; }
  }));
  $("#advanced-toggle").addEventListener("click", () => {
    const open = $("#advanced-controls").classList.toggle("open");
    $("#advanced-toggle").textContent = `LLM + ADVANCED CONTROLS ${open ? "-" : "+"}`;
    $("#advanced-toggle").setAttribute("aria-expanded", String(open));
  });
  $("#llm-mode").addEventListener("change", updateLLMControls);
  $("#llm-provider").addEventListener("change", updateLLMControls);
  $("#llm-key-toggle").addEventListener("click", () => {
    const key = $("#llm-api-key");
    const visible = key.type === "text";
    key.type = visible ? "password" : "text";
    $("#llm-key-toggle").textContent = visible ? "SHOW" : "HIDE";
  });
  $("#mission-form").addEventListener("submit", submitMission);
  $("#run-search").addEventListener("input", renderRuns);
  $("#pipeline-run").addEventListener("change", () => loadInsights("pipeline"));
  $("#explain-run").addEventListener("change", () => loadInsights("explain"));
  $("#monitor-run").addEventListener("change", loadMonitoring);
  $("#refresh-monitor").addEventListener("click", loadMonitoring);
  $("#observe-button").addEventListener("click", observeBatch);
  $("#incremental-button").addEventListener("click", evaluateIncrementalUpdate);
  $("#what-if-button").addEventListener("click", evaluateWhatIf);
  $("#refresh-audit").addEventListener("click", loadAudit);
  bindCanvasRotation($("#pipeline-canvas"), state.pipelineRotation, drawPipeline);
  bindCanvasRotation($("#geometry-canvas"), state.geometryRotation, drawGeometry);
  $("#geometry-canvas").addEventListener("pointermove", updateGeometryHover);
  $("#geometry-canvas").addEventListener("pointerleave", () => {
    state.geometryView.hoveredRow = null;
    $("#geometry-tooltip").hidden = true;
    drawGeometry();
  });
  $("#geometry-canvas").addEventListener("wheel", (event) => {
    event.preventDefault();
    state.geometryView.zoom = Math.max(.55, Math.min(2.4, state.geometryView.zoom * (event.deltaY < 0 ? 1.1 : .9)));
    drawGeometry();
  }, { passive: false });
  $("#geometry-color").addEventListener("change", (event) => {
    state.geometryView.colorMode = event.target.value;
    state.geometryView.hiddenCategories.clear();
    state.geometryView.legendKey = "";
    drawGeometry();
  });
  $("#geometry-legend").addEventListener("click", (event) => {
    const filter = event.target.closest("[data-geometry-category]");
    if (!filter) return;
    const category = filter.dataset.geometryCategory;
    if (state.geometryView.hiddenCategories.has(category)) state.geometryView.hiddenCategories.delete(category);
    else state.geometryView.hiddenCategories.add(category);
    state.geometryView.legendKey = "";
    drawGeometry();
  });
  $("#geometry-point-size").addEventListener("input", (event) => {
    state.geometryView.pointSize = Number(event.target.value) / 100;
    drawGeometry();
  });
  $("#geometry-surface").addEventListener("change", (event) => {
    state.geometryView.showSurface = event.target.checked;
    drawGeometry();
  });
  $("#geometry-reset").addEventListener("click", () => resetGeometryView("isometric"));
  $("#geometry-top").addEventListener("click", () => resetGeometryView("top"));
  window.addEventListener("resize", () => { drawPipeline(); drawGeometry(); });
  document.addEventListener("click", (event) => {
    const documentCard = event.target.closest("[data-document]");
    if (documentCard?.classList.contains("unavailable")) {
      event.preventDefault();
      toast("This PDF has not been uploaded yet.", "error");
      return;
    }
    const artifact = event.target.closest("[data-artifact-run]");
    if (artifact) {
      downloadArtifact(artifact.dataset.artifactRun, artifact.dataset.artifactName);
      return;
    }
    const deploy = event.target.closest("[data-deploy-run]");
    if (deploy) { deployRun(deploy.dataset.deployRun); return; }
    const undeploy = event.target.closest("[data-undeploy-run]");
    if (undeploy) { deployRun(undeploy.dataset.undeployRun, true); return; }
    const run = event.target.closest("[data-run-id]");
    if (run) openRun(run.dataset.runId);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && event.target.matches("[data-run-id]")) openRun(event.target.dataset.runId);
  });
  $("#dialog-close").addEventListener("click", () => { state.selectedRun = null; $("#run-dialog").close(); });
  $("#run-dialog").addEventListener("click", (event) => {
    if (event.target === $("#run-dialog")) { state.selectedRun = null; $("#run-dialog").close(); }
  });
}

async function boot() {
  bindEvents();
  updateLLMControls();
  updateComputeControls();
  await loadDocuments();
  try {
    state.auth.config = await fetch(apiUrl("/api/auth/config", "cloud")).then((response) => response.json());
    configureDatasetSources();
    if (state.auth.config.required) {
      try {
        const saved = JSON.parse(sessionStorage.getItem("autonexus.firebase") || "null");
        if (saved) Object.assign(state.auth, saved);
      } catch (_) { sessionStorage.removeItem("autonexus.firebase"); }
      if (!state.auth.idToken) {
        showAuthGate();
      } else {
        try { state.auth.user = await api("/api/auth/me", {}, "cloud"); hideAuthGate(); }
        catch (_) { signOut(); }
      }
    } else {
      state.auth.user = await api("/api/auth/me", {}, "cloud");
      hideAuthGate();
    }
    renderIdentity();
  } catch (error) {
    showAuthGate(`Authentication configuration failed: ${error.message}`);
  }
  const initialView = location.hash.slice(1);
  if (["overview", "new-mission", "runs", "pipeline", "explain", "monitor", "audit"].includes(initialView)) navigate(initialView, false);
  try {
    state.health = await api("/api/health", {}, "cloud");
    $("#engine-chip").classList.add("online");
    $("#engine-label").textContent = "ENGINE ONLINE";
    $("#version-label").textContent = `v${state.health.version}`;
    $("#metric-engine").textContent = "ONLINE";
  } catch (error) {
    $("#engine-label").textContent = "ENGINE OFFLINE";
    $("#metric-engine").textContent = "OFFLINE";
    toast(error.message, "error");
  }
  if (state.auth.user) await refreshRuns();
  window.setInterval(() => { if (state.auth.user) refreshRuns(); }, 3000);
}

document.addEventListener("DOMContentLoaded", boot);
