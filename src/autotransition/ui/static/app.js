const state = {
  presets: [],
  models: [],
  selectedPreset: null,
  selectedModel: null,
  sourceProbe: null,
  toastTimer: null,
  generationPollTimer: null,
  isGenerating: false,
  advancedDirty: false,
  generatedResults: [],
};

const el = {
  ffmpegBadge: document.querySelector("#ffmpegBadge"),
  modelCountBadge: document.querySelector("#modelCountBadge"),
  runtimeBadge: document.querySelector("#runtimeBadge"),
  sourceState: document.querySelector("#sourceState"),
  actionState: document.querySelector("#actionState"),
  systemState: document.querySelector("#systemState"),
  runtimeState: document.querySelector("#runtimeState"),
  modelState: document.querySelector("#modelState"),
  presetSelect: document.querySelector("#presetSelect"),
  presetSummary: document.querySelector("#presetSummary"),
  captionInput: document.querySelector("#captionInput"),
  sourcePath: document.querySelector("#sourcePath"),
  sourceFile: document.querySelector("#sourceFile"),
  selectedFileName: document.querySelector("#selectedFileName"),
  loadSourceButton: document.querySelector("#loadSourceButton"),
  sourceDuration: document.querySelector("#sourceDuration"),
  sourceFormatReadout: document.querySelector("#sourceFormatReadout"),
  outputFormatReadout: document.querySelector("#outputFormatReadout"),
  sourceAudio: document.querySelector("#sourceAudio"),
  currentTimeReadout: document.querySelector("#currentTimeReadout"),
  continuationReadout: document.querySelector("#continuationReadout"),
  continuationSlider: document.querySelector("#continuationSlider"),
  contextRange: document.querySelector("#contextRange"),
  futureRange: document.querySelector("#futureRange"),
  outputDir: document.querySelector("#outputDir"),
  contextSeconds: document.querySelector("#contextSeconds"),
  newSeconds: document.querySelector("#newSeconds"),
  bpmInput: document.querySelector("#bpmInput"),
  keyInput: document.querySelector("#keyInput"),
  seedInput: document.querySelector("#seedInput"),
  inferenceSteps: document.querySelector("#inferenceSteps"),
  guidanceScale: document.querySelector("#guidanceScale"),
  shiftValue: document.querySelector("#shiftValue"),
  resetAceDefaultsButton: document.querySelector("#resetAceDefaultsButton"),
  generateButton: document.querySelector("#generateButton"),
  generationActivity: document.querySelector("#generationActivity"),
  refreshButton: document.querySelector("#refreshButton"),
  generatedList: document.querySelector("#generatedList"),
  modelSelect: document.querySelector("#modelSelect"),
  modelDetails: document.querySelector("#modelDetails"),
  autoInstallModel: document.querySelector("#autoInstallModel"),
  installModelButton: document.querySelector("#installModelButton"),
  systemStatus: document.querySelector("#systemStatus"),
  runtimeDetails: document.querySelector("#runtimeDetails"),
  copyRuntimeCommandButton: document.querySelector("#copyRuntimeCommandButton"),
  logList: document.querySelector("#logList"),
  toast: document.querySelector("#toast"),
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = body && body.detail ? body.detail : `Request failed: ${response.status}`;
    throw new Error(detail);
  }
  return body;
}

function setPill(node, text, tone = "neutral") {
  node.textContent = text;
  node.className = node.className
    .split(" ")
    .filter((part) => !["ok", "warn", "error", "neutral"].includes(part))
    .join(" ");
  node.classList.add(tone);
}

function showToast(message) {
  el.toast.textContent = message;
  el.toast.classList.add("visible");
  window.clearTimeout(state.toastTimer);
  state.toastTimer = window.setTimeout(() => {
    el.toast.classList.remove("visible");
  }, 3600);
}

function formatTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const whole = Math.floor(seconds);
  const mins = Math.floor(whole / 60);
  const secs = String(whole % 60).padStart(2, "0");
  return `${mins}:${secs}`;
}

function option(label, value) {
  const item = document.createElement("option");
  item.value = value;
  item.textContent = label;
  return item;
}

function applyPreset(preset) {
  state.selectedPreset = preset;
  el.captionInput.value = preset.caption;
  el.contextSeconds.value = preset.config.context_seconds;
  el.newSeconds.value = preset.config.new_section_seconds;
  el.presetSummary.innerHTML = `<strong>${preset.name}</strong><br>${preset.description}`;
  updateSelectionReadout();
}

function renderPresets() {
  el.presetSelect.replaceChildren();
  state.presets.forEach((preset) => {
    el.presetSelect.appendChild(option(preset.name, preset.slug));
  });
  if (state.presets.length) {
    applyPreset(state.presets[0]);
  }
}

function modelTone(model) {
  return model.status.state === "ready" ? "ok" : "warn";
}

function renderModels() {
  el.modelSelect.replaceChildren();
  state.models.forEach((model) => {
    const label = `${model.display_name} (${model.status.state})`;
    el.modelSelect.appendChild(option(label, model.slug));
  });
  if (state.models.length) {
    applyModel(state.models[0]);
  }
}

function applyModel(model) {
  state.selectedModel = model;
  setPill(el.modelState, model.status.state.replace("_", " "), modelTone(model));
  el.modelDetails.innerHTML = [
    `<strong>${model.display_name}</strong>`,
    `${model.quality_label} quality, ${model.speed_label.toLowerCase()} profile`,
    model.vram_guidance,
    `Repo: ${model.repo_id}`,
    `Local: ${model.status.local_path}`,
  ].join("<br>");
  el.installModelButton.disabled = model.status.state === "ready";
  if (!state.advancedDirty) {
    applyAceDefaults(model);
  }
}

function setNumeric(node, value) {
  node.value = value === null || value === undefined ? "" : String(value);
}

function applyAceDefaults(model) {
  const defaults = (model && (model.generation_defaults || model.repaint_defaults)) || {};
  setNumeric(el.inferenceSteps, defaults.inference_steps);
  setNumeric(el.guidanceScale, defaults.guidance_scale);
  setNumeric(el.shiftValue, defaults.shift);
  state.advancedDirty = false;
}

function renderStatus(status) {
  setPill(el.ffmpegBadge, status.ffmpeg_available ? "ffmpeg ready" : "ffmpeg missing", status.ffmpeg_available ? "ok" : "error");
  setPill(el.modelCountBadge, `${status.repaint_model_count} ACE models`, "ok");
  setPill(el.runtimeBadge, `Python ${status.python_version}`, "neutral");
  setPill(el.systemState, "Live", "ok");
  el.systemStatus.innerHTML = `
    <dt>Python</dt><dd>${status.python_version}</dd>
    <dt>ffmpeg</dt><dd>${status.ffmpeg_path || "Not found"}</dd>
    <dt>Inputs</dt><dd>${(status.supported_input_formats || []).join(", ")}</dd>
    <dt>Output</dt><dd>${String(status.default_scaffold_format || "wav").toUpperCase()} scaffold</dd>
    <dt>Models</dt><dd>${status.models_dir}</dd>
    <dt>Folder</dt><dd>${status.cwd}</dd>
  `;
  el.outputFormatReadout.textContent = `Output scaffold: ${String(status.default_scaffold_format || "wav").toUpperCase()}`;
}

function renderRuntime(runtime) {
  const tone = runtime.api_running ? "ok" : runtime.installed ? "warn" : "error";
  setPill(el.runtimeState, runtime.api_running ? "API running" : runtime.installed ? "Installed" : "Not installed", tone);
  el.runtimeDetails.innerHTML = [
    `<strong>${runtime.message}</strong>`,
    `Install: ${runtime.install_dir}`,
    `API: ${runtime.api_url}`,
    `uv: ${runtime.uv_available ? "available" : "missing"}`,
    `git: ${runtime.git_available ? "available" : "missing"}`,
    `Setup: ${runtime.simple_setup_command}`,
    `Start: ${runtime.simple_start_command}`,
  ].join("<br>");
  el.copyRuntimeCommandButton.dataset.command = `${runtime.simple_setup_command}\n${runtime.simple_start_command}`;
}

function renderLogs(logs) {
  el.logList.replaceChildren();
  logs.forEach((entry) => {
    const item = document.createElement("li");
    const level = document.createElement("span");
    level.className = `level ${entry.level}`;
    level.textContent = entry.level;
    const text = document.createTextNode(`${entry.timestamp} ${entry.message}`);
    item.append(level, text);
    el.logList.appendChild(item);
  });
}

function activityTone(phase) {
  if (phase === "error") return "error";
  if (["downloading", "initializing", "generating"].includes(phase)) return "warn";
  if (phase === "ready" || phase === "complete") return "ok";
  return "neutral";
}

function activityLabel(activity) {
  const phase = activity.phase || "idle";
  if (phase === "downloading") return "Downloading";
  if (phase === "initializing") return "Initializing";
  if (phase === "generating") return "Generating";
  if (phase === "error") return "Runtime error";
  if (phase === "ready") return "Runtime ready";
  return "Waiting";
}

function renderActivity(activity) {
  const message = activity.message || "No ACE-Step activity yet.";
  const detail = activity.detail ? `<br>${activity.detail}` : "";
  el.generationActivity.innerHTML = `<strong>${activityLabel(activity)}</strong><br>${message}${detail}`;
  if (state.isGenerating) {
    setPill(el.actionState, activityLabel(activity), activityTone(activity.phase));
  }
}

async function refreshActivity() {
  const activity = await api("/api/runtime/activity");
  renderActivity(activity);
  return activity;
}

function startGenerationPolling() {
  stopGenerationPolling();
  state.isGenerating = true;
  refreshActivity().catch(() => {});
  state.generationPollTimer = window.setInterval(() => {
    Promise.all([refreshActivity(), refreshLogs()]).catch(() => {});
  }, 2500);
}

function stopGenerationPolling() {
  state.isGenerating = false;
  if (state.generationPollTimer) {
    window.clearInterval(state.generationPollTimer);
    state.generationPollTimer = null;
  }
}

function renderGeneratedList() {
  el.generatedList.replaceChildren();
  if (!state.generatedResults.length) {
    const empty = document.createElement("div");
    empty.className = "empty-result";
    empty.textContent = "No generated audio yet.";
    el.generatedList.appendChild(empty);
    return;
  }

  state.generatedResults.forEach((item, index) => {
    const { result, plan } = item;
    const row = document.createElement("article");
    row.className = "generated-item";
    const outputPath = result.generated_audio_path || "";
    const audio = outputPath
      ? `<audio controls preload="metadata" src="/api/audio?path=${encodeURIComponent(outputPath)}"></audio>`
      : `<div class="empty-result">No playable audio for this result.</div>`;
    row.innerHTML = `
      <div class="generated-title">
        <strong>${index === 0 ? "Latest" : "Result"} - ${result.status}</strong>
        <span>${result.model_slug || "model"}</span>
      </div>
      ${audio}
      <dl class="path-list">
        <dt>Message</dt><dd>${result.message}</dd>
        <dt>Mode</dt><dd>${plan.generation_region === "repaint_existing" ? "Repaint existing audio" : "Extend after marker"}</dd>
        <dt>Source</dt><dd>${formatTime(plan.tail_start_seconds)} to ${formatTime(plan.tail_end_seconds)}</dd>
        <dt>Generated</dt><dd>${Number(plan.new_section_seconds || 0).toFixed(1)}s</dd>
        <dt>Output</dt><dd>${outputPath || "None"}</dd>
        <dt>Metadata</dt><dd>${result.generated_metadata_path || result.scaffold_metadata_path}</dd>
        <dt>Prompt</dt><dd>${plan.caption}</dd>
      </dl>
    `;
    el.generatedList.appendChild(row);
  });
}

function addGeneratedResult(result, plan) {
  state.generatedResults.unshift({ result, plan });
  state.generatedResults = state.generatedResults.slice(0, 12);
  renderGeneratedList();
}

async function loadAll() {
  const [status, runtime, presets, models, logs] = await Promise.all([
    api("/api/status"),
    api("/api/runtime/status"),
    api("/api/presets"),
    api("/api/models"),
    api("/api/logs"),
  ]);
  state.presets = presets;
  state.models = models;
  renderStatus(status);
  renderRuntime(runtime);
  renderPresets();
  renderModels();
  renderLogs(logs);
}

function numericValue(node) {
  return node.value === "" ? null : Number(node.value);
}

function currentSettings() {
  return {
    contextSeconds: numericValue(el.contextSeconds),
    newSeconds: numericValue(el.newSeconds),
  };
}

function updateSelectionReadout() {
  const continuation = Number(el.continuationSlider.value || 0);
  const settings = currentSettings();
  const context = settings.contextSeconds || 0;
  const future = settings.newSeconds || 0;
  const tail = context;
  const start = continuation - tail;
  el.continuationReadout.textContent = `Continue at ${formatTime(continuation)}`;
  el.futureRange.textContent = `Generate new section: ${future.toFixed(1)}s`;
  if (!state.sourceProbe) {
    el.contextRange.textContent = "Context not selected";
    return;
  }
  if (start < 0) {
    el.contextRange.textContent = `${tail.toFixed(1)}s source context needs marker at ${formatTime(tail)} or later`;
    setPill(el.sourceState, "Marker too early", "warn");
    return;
  }
  el.contextRange.textContent = `Source context: ${formatTime(start)} to ${formatTime(continuation)} (${tail.toFixed(1)}s)`;
  setPill(el.sourceState, "Source loaded", "ok");
}

function aceStepSettingsPayload() {
  return {
    inference_steps: numericValue(el.inferenceSteps),
    guidance_scale: numericValue(el.guidanceScale),
    shift: numericValue(el.shiftValue),
  };
}

async function loadProbeIntoPlayer(sourcePath, probe) {
  state.sourceProbe = probe;
  el.sourcePath.value = sourcePath;
  el.sourceAudio.src = `/api/source/audio?path=${encodeURIComponent(sourcePath)}`;
  el.continuationSlider.max = String(probe.duration_seconds);
  el.continuationSlider.value = String(Math.max(0, probe.duration_seconds - 1));
  el.sourceDuration.textContent = `Duration ${formatTime(probe.duration_seconds)}`;
  el.sourceFormatReadout.textContent = `Source format: ${probe.source_format}; decoded in background`;
  setPill(el.sourceState, "Source loaded", "ok");
  updateSelectionReadout();
}

async function loadSource() {
  setPill(el.sourceState, "Loading", "warn");
  el.loadSourceButton.disabled = true;
  try {
    const sourcePath = el.sourcePath.value.trim();
    const probe = await api("/api/source/probe", {
      method: "POST",
      body: JSON.stringify({ source_path: sourcePath }),
    });
    await loadProbeIntoPlayer(sourcePath, probe);
    showToast("Source loaded");
  } catch (error) {
    state.sourceProbe = null;
    setPill(el.sourceState, "Error", "error");
    showToast(error.message);
  } finally {
    el.loadSourceButton.disabled = false;
    refreshLogs();
  }
}

async function uploadSourceFile() {
  const file = el.sourceFile.files && el.sourceFile.files[0];
  if (!file) return;

  setPill(el.sourceState, "Uploading", "warn");
  el.selectedFileName.textContent = file.name;
  el.loadSourceButton.disabled = true;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/api/source/upload", {
      method: "POST",
      body: formData,
    });
    const body = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(body && body.detail ? body.detail : `Upload failed: ${response.status}`);
    }

    await loadProbeIntoPlayer(body.stored_path, body.probe);
    showToast("Audio file loaded");
  } catch (error) {
    state.sourceProbe = null;
    setPill(el.sourceState, "Error", "error");
    showToast(error.message);
  } finally {
    el.loadSourceButton.disabled = false;
    refreshLogs();
  }
}

async function generateTransition() {
  setPill(el.actionState, "Generating", "warn");
  el.generationActivity.innerHTML = "<strong>Starting</strong><br>Preparing source selection and ACE-Step request.";
  el.generateButton.disabled = true;
  startGenerationPolling();
  try {
    const payload = {
      source_path: el.sourcePath.value.trim(),
      continuation_point_seconds: Number(el.continuationSlider.value || 0),
      generation_region: "extend",
      preset: el.presetSelect.value,
      model_slug: state.selectedModel ? state.selectedModel.slug : el.modelSelect.value,
      auto_install: el.autoInstallModel.checked,
      caption: el.captionInput.value.trim(),
      output_dir: el.outputDir.value.trim() || null,
      context_seconds: numericValue(el.contextSeconds),
      new_section_seconds: numericValue(el.newSeconds),
      bpm: numericValue(el.bpmInput),
      key: el.keyInput.value.trim() || null,
      seed: numericValue(el.seedInput),
      ace_step: aceStepSettingsPayload(),
    };
    const response = await api("/api/generate/from-selection", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    addGeneratedResult(response.result, response.plan);
    if (response.result.status === "complete") {
      setPill(el.actionState, "Complete", "ok");
      el.generationActivity.innerHTML = "<strong>Complete</strong><br>Transition generated.";
      showToast("Transition generated");
    } else {
      setPill(el.actionState, "Needs runtime", "warn");
      el.generationActivity.innerHTML = `<strong>Stopped</strong><br>${response.result.message}`;
      showToast(response.result.message);
    }
  } catch (error) {
    setPill(el.actionState, "Error", "error");
    el.generationActivity.innerHTML = `<strong>Error</strong><br>${error.message}`;
    showToast(error.message);
  } finally {
    stopGenerationPolling();
    el.generateButton.disabled = false;
    refreshLogs();
  }
}

async function installModel() {
  if (!state.selectedModel) return;
  setPill(el.modelState, "downloading", "warn");
  el.installModelButton.disabled = true;
  try {
    await api(`/api/models/${state.selectedModel.slug}/install`, { method: "POST" });
    showToast("Model installed");
    await refreshModels();
  } catch (error) {
    setPill(el.modelState, "failed", "error");
    showToast(error.message);
  } finally {
    refreshLogs();
  }
}

async function refreshModels() {
  state.models = await api("/api/models");
  const selectedSlug = el.modelSelect.value || (state.selectedModel && state.selectedModel.slug);
  renderModels();
  const selected = state.models.find((model) => model.slug === selectedSlug);
  if (selected) {
    el.modelSelect.value = selected.slug;
    applyModel(selected);
  }
}

async function refreshLogs() {
  renderLogs(await api("/api/logs"));
}

async function refreshStatus() {
  renderStatus(await api("/api/status"));
  renderRuntime(await api("/api/runtime/status"));
  await refreshActivity();
  await refreshModels();
  await refreshLogs();
  showToast("Status refreshed");
}

el.presetSelect.addEventListener("change", () => {
  const preset = state.presets.find((item) => item.slug === el.presetSelect.value);
  if (preset) applyPreset(preset);
});

el.modelSelect.addEventListener("change", () => {
  const model = state.models.find((item) => item.slug === el.modelSelect.value);
  if (model) applyModel(model);
});

el.generateButton.addEventListener("click", generateTransition);
el.loadSourceButton.addEventListener("click", loadSource);
el.sourceFile.addEventListener("change", uploadSourceFile);
el.installModelButton.addEventListener("click", installModel);
el.refreshButton.addEventListener("click", refreshStatus);
el.copyRuntimeCommandButton.addEventListener("click", async () => {
  const command = el.copyRuntimeCommandButton.dataset.command || "";
  await navigator.clipboard.writeText(command);
  showToast("Setup commands copied");
});
el.continuationSlider.addEventListener("input", updateSelectionReadout);
el.sourceAudio.addEventListener("timeupdate", () => {
  el.currentTimeReadout.textContent = formatTime(el.sourceAudio.currentTime);
});
el.sourceAudio.addEventListener("seeked", () => {
  el.currentTimeReadout.textContent = formatTime(el.sourceAudio.currentTime);
});

[el.contextSeconds, el.newSeconds].forEach((node) => {
  node.addEventListener("input", updateSelectionReadout);
});

[
  el.inferenceSteps,
  el.guidanceScale,
  el.shiftValue,
].forEach((node) => {
  node.addEventListener("input", () => {
    state.advancedDirty = true;
  });
  node.addEventListener("change", () => {
    state.advancedDirty = true;
  });
});

el.resetAceDefaultsButton.addEventListener("click", () => {
  if (state.selectedModel) {
    applyAceDefaults(state.selectedModel);
    showToast("ACE-Step defaults restored");
  }
});

loadAll().catch((error) => {
  setPill(el.actionState, "Error", "error");
  showToast(error.message);
});

window.setInterval(refreshLogs, 5000);
