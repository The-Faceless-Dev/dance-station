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
  musicResults: [],
  instrumentClips: [],
  instrumentBank: [],
  instrumentTracks: [
    {
      id: "track-main",
      label: "Track 1",
      kind: "instrument",
      instrument: "synth.lead",
      volume: 0.85,
      pan: 0,
      muted: false,
      playDuringRecord: true,
      notes: [],
    },
  ],
  activeInstrumentTrackId: "track-main",
  selectedInstrumentNoteId: null,
  selectedInstrumentNoteIds: [],
  instrumentCursorBeat: 0,
  instrumentRecording: false,
  instrumentAudioContext: null,
  instrumentPlayingSources: [],
  instrumentPlaybackId: 0,
  instrumentTransportStartTime: null,
  instrumentTransportStartBeat: 0,
  instrumentCountdownTimer: null,
  instrumentAudioBufferCache: new Map(),
  instrumentSampleBufferCache: new Map(),
  pianoRollView: {
    beatOffset: 0,
    visibleBeats: 16,
    pitchOffset: 36,
    visiblePitches: 49,
  },
  instrumentPreviewUrl: null,
  instrumentDrag: null,
  instrumentNoteClipboard: [],
  extractionTracks: [],
  extractionResults: [],
  editorAssets: [],
  selectedEditorAsset: null,
  extractSourceProbe: null,
};

const el = {
  transitionTabButton: document.querySelector("#transitionTabButton"),
  extractionTabButton: document.querySelector("#extractionTabButton"),
  musicTabButton: document.querySelector("#musicTabButton"),
  instrumentLabTabButton: document.querySelector("#instrumentLabTabButton"),
  audioEditTabButton: document.querySelector("#audioEditTabButton"),
  transitionPage: document.querySelector("#transitionPage"),
  extractionPage: document.querySelector("#extractionPage"),
  musicPage: document.querySelector("#musicPage"),
  instrumentLabPage: document.querySelector("#instrumentLabPage"),
  audioEditPage: document.querySelector("#audioEditPage"),
  ffmpegBadge: document.querySelector("#ffmpegBadge"),
  modelCountBadge: document.querySelector("#modelCountBadge"),
  runtimeBadge: document.querySelector("#runtimeBadge"),
  sourceState: document.querySelector("#sourceState"),
  actionState: document.querySelector("#actionState"),
  systemState: document.querySelector("#systemState"),
  runtimeState: document.querySelector("#runtimeState"),
  modelState: document.querySelector("#modelState"),
  promptSummary: document.querySelector("#promptSummary"),
  captionInput: document.querySelector("#captionInput"),
  sourcePath: document.querySelector("#sourcePath"),
  sourceFile: document.querySelector("#sourceFile"),
  selectedFileName: document.querySelector("#selectedFileName"),
  sourceAssetSelect: document.querySelector("#sourceAssetSelect"),
  loadSourceAssetButton: document.querySelector("#loadSourceAssetButton"),
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
  repaintOverlapSeconds: document.querySelector("#repaintOverlapSeconds"),
  bpmInput: document.querySelector("#bpmInput"),
  keyInput: document.querySelector("#keyInput"),
  seedInput: document.querySelector("#seedInput"),
  inferenceSteps: document.querySelector("#inferenceSteps"),
  guidanceScale: document.querySelector("#guidanceScale"),
  shiftValue: document.querySelector("#shiftValue"),
  repaintStrength: document.querySelector("#repaintStrength"),
  repaintMode: document.querySelector("#repaintMode"),
  repaintLatentCrossfadeFrames: document.querySelector("#repaintLatentCrossfadeFrames"),
  repaintWavCrossfadeSec: document.querySelector("#repaintWavCrossfadeSec"),
  resetAceDefaultsButton: document.querySelector("#resetAceDefaultsButton"),
  generateButton: document.querySelector("#generateButton"),
  generationActivity: document.querySelector("#generationActivity"),
  refreshButton: document.querySelector("#refreshButton"),
  generatedList: document.querySelector("#generatedList"),
  modelDetails: document.querySelector("#modelDetails"),
  autoInstallModel: document.querySelector("#autoInstallModel"),
  installModelButton: document.querySelector("#installModelButton"),
  systemStatus: document.querySelector("#systemStatus"),
  runtimeDetails: document.querySelector("#runtimeDetails"),
  copyRuntimeCommandButton: document.querySelector("#copyRuntimeCommandButton"),
  logList: document.querySelector("#logList"),
  extractSourceState: document.querySelector("#extractSourceState"),
  extractSourceFile: document.querySelector("#extractSourceFile"),
  extractSelectedFileName: document.querySelector("#extractSelectedFileName"),
  extractSourceAssetSelect: document.querySelector("#extractSourceAssetSelect"),
  loadExtractSourceAssetButton: document.querySelector("#loadExtractSourceAssetButton"),
  extractSourcePath: document.querySelector("#extractSourcePath"),
  loadExtractSourceButton: document.querySelector("#loadExtractSourceButton"),
  extractSourceDuration: document.querySelector("#extractSourceDuration"),
  extractSourceAudio: document.querySelector("#extractSourceAudio"),
  extractSourceFormatReadout: document.querySelector("#extractSourceFormatReadout"),
  extractTrackSelect: document.querySelector("#extractTrackSelect"),
  extractLabelInput: document.querySelector("#extractLabelInput"),
  extractOutputFormat: document.querySelector("#extractOutputFormat"),
  extractSeedInput: document.querySelector("#extractSeedInput"),
  extractInferenceSteps: document.querySelector("#extractInferenceSteps"),
  extractGuidanceScale: document.querySelector("#extractGuidanceScale"),
  extractShift: document.querySelector("#extractShift"),
  extractInstruction: document.querySelector("#extractInstruction"),
  runExtractionButton: document.querySelector("#runExtractionButton"),
  refreshExtractionsButton: document.querySelector("#refreshExtractionsButton"),
  extractActionState: document.querySelector("#extractActionState"),
  mergeLabelInput: document.querySelector("#mergeLabelInput"),
  mergeOutputFormat: document.querySelector("#mergeOutputFormat"),
  mergeExtractionsButton: document.querySelector("#mergeExtractionsButton"),
  extractionActivity: document.querySelector("#extractionActivity"),
  extractionList: document.querySelector("#extractionList"),
  extractRuntimeState: document.querySelector("#extractRuntimeState"),
  extractLogList: document.querySelector("#extractLogList"),
  musicActionState: document.querySelector("#musicActionState"),
  musicModelState: document.querySelector("#musicModelState"),
  musicPrompt: document.querySelector("#musicPrompt"),
  musicInstrumental: document.querySelector("#musicInstrumental"),
  musicVocalLanguage: document.querySelector("#musicVocalLanguage"),
  musicLyrics: document.querySelector("#musicLyrics"),
  musicLabelInput: document.querySelector("#musicLabelInput"),
  musicModelSelect: document.querySelector("#musicModelSelect"),
  musicOutputFormat: document.querySelector("#musicOutputFormat"),
  musicDuration: document.querySelector("#musicDuration"),
  musicSeed: document.querySelector("#musicSeed"),
  musicInferenceSteps: document.querySelector("#musicInferenceSteps"),
  musicGuidanceScale: document.querySelector("#musicGuidanceScale"),
  musicShift: document.querySelector("#musicShift"),
  musicInferMethod: document.querySelector("#musicInferMethod"),
  musicUseTiledDecode: document.querySelector("#musicUseTiledDecode"),
  musicDcwEnabled: document.querySelector("#musicDcwEnabled"),
  musicVelocityNormThreshold: document.querySelector("#musicVelocityNormThreshold"),
  musicVelocityEmaFactor: document.querySelector("#musicVelocityEmaFactor"),
  runMusicButton: document.querySelector("#runMusicButton"),
  refreshMusicButton: document.querySelector("#refreshMusicButton"),
  musicActivity: document.querySelector("#musicActivity"),
  musicList: document.querySelector("#musicList"),
  musicLogList: document.querySelector("#musicLogList"),
  instrumentLabState: document.querySelector("#instrumentLabState"),
  instrumentClipLabel: document.querySelector("#instrumentClipLabel"),
  instrumentBpm: document.querySelector("#instrumentBpm"),
  instrumentKey: document.querySelector("#instrumentKey"),
  instrumentBars: document.querySelector("#instrumentBars"),
  instrumentOctave: document.querySelector("#instrumentOctave"),
  addInstrumentTrackButton: document.querySelector("#addInstrumentTrackButton"),
  instrumentTrackList: document.querySelector("#instrumentTrackList"),
  instrumentAssetSelect: document.querySelector("#instrumentAssetSelect"),
  importInstrumentAssetButton: document.querySelector("#importInstrumentAssetButton"),
  instrumentActiveTrackReadout: document.querySelector("#instrumentActiveTrackReadout"),
  playInstrumentButton: document.querySelector("#playInstrumentButton"),
  stopInstrumentButton: document.querySelector("#stopInstrumentButton"),
  recordInstrumentButton: document.querySelector("#recordInstrumentButton"),
  deleteInstrumentNoteButton: document.querySelector("#deleteInstrumentNoteButton"),
  copyInstrumentNotesButton: document.querySelector("#copyInstrumentNotesButton"),
  pasteInstrumentNotesButton: document.querySelector("#pasteInstrumentNotesButton"),
  instrumentPianoRoll: document.querySelector("#instrumentPianoRoll"),
  pianoRollScroll: document.querySelector("#pianoRollScroll"),
  pianoRollZoomOutButton: document.querySelector("#pianoRollZoomOutButton"),
  pianoRollZoomInButton: document.querySelector("#pianoRollZoomInButton"),
  pianoRollFitButton: document.querySelector("#pianoRollFitButton"),
  pianoRollViewportReadout: document.querySelector("#pianoRollViewportReadout"),
  instrumentPianoKeys: document.querySelector("#instrumentPianoKeys"),
  instrumentPatch: document.querySelector("#instrumentPatch"),
  instrumentBankState: document.querySelector("#instrumentBankState"),
  instrumentInfo: document.querySelector("#instrumentInfo"),
  sfzInstrumentLabel: document.querySelector("#sfzInstrumentLabel"),
  sfzInstrumentFile: document.querySelector("#sfzInstrumentFile"),
  sfzSampleFiles: document.querySelector("#sfzSampleFiles"),
  importSfzButton: document.querySelector("#importSfzButton"),
  instrumentMasterVolume: document.querySelector("#instrumentMasterVolume"),
  instrumentRenderState: document.querySelector("#instrumentRenderState"),
  renderInstrumentButton: document.querySelector("#renderInstrumentButton"),
  saveInstrumentTrackButton: document.querySelector("#saveInstrumentTrackButton"),
  saveInstrumentButton: document.querySelector("#saveInstrumentButton"),
  instrumentPreviewAudio: document.querySelector("#instrumentPreviewAudio"),
  instrumentClipList: document.querySelector("#instrumentClipList"),
  editorAssetState: document.querySelector("#editorAssetState"),
  editorAssetSearch: document.querySelector("#editorAssetSearch"),
  editorCategoryFilter: document.querySelector("#editorCategoryFilter"),
  refreshEditorAssetsButton: document.querySelector("#refreshEditorAssetsButton"),
  editorAssetList: document.querySelector("#editorAssetList"),
  editorCurrentAsset: document.querySelector("#editorCurrentAsset"),
  audioEditorFrame: document.querySelector("#audioEditorFrame"),
  reloadAudioEditorButton: document.querySelector("#reloadAudioEditorButton"),
  openAudioEditorButton: document.querySelector("#openAudioEditorButton"),
  editSaveLabelInput: document.querySelector("#editSaveLabelInput"),
  editSaveFile: document.querySelector("#editSaveFile"),
  editSaveFileName: document.querySelector("#editSaveFileName"),
  editSourceAssetReadout: document.querySelector("#editSourceAssetReadout"),
  editSaveState: document.querySelector("#editSaveState"),
  saveEditButton: document.querySelector("#saveEditButton"),
  toast: document.querySelector("#toast"),
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = formatApiDetail(body && body.detail ? body.detail : `Request failed: ${response.status}`);
    throw new Error(detail);
  }
  return body;
}

function formatApiDetail(detail) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (typeof item === "string") return item;
        const location = Array.isArray(item.loc) ? item.loc.join(".") : "request";
        return `${location}: ${item.msg || JSON.stringify(item)}`;
      })
      .join("; ");
  }
  if (detail && typeof detail === "object") return detail.message || JSON.stringify(detail);
  return String(detail);
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

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function applyPreset(preset) {
  state.selectedPreset = preset;
  if (!el.captionInput.value.trim()) el.captionInput.value = preset.caption;
  el.contextSeconds.value = preset.config.context_seconds;
  el.newSeconds.value = preset.config.new_section_seconds;
  el.repaintOverlapSeconds.value = preset.config.repaint_overlap_seconds;
  if (!el.bpmInput.value) el.bpmInput.value = "120";
  updateSelectionReadout();
}

function renderPresets() {
  if (state.presets.length) {
    applyPreset(state.presets[0]);
  }
}

function modelTone(model) {
  return model.status.state === "ready" ? "ok" : "warn";
}

function renderModels() {
  if (state.models.length) {
    const preferred =
      state.models.find((model) => model.slug === "acestep-v15-turbo" && model.status.state === "ready") ||
      state.models.find((model) => model.slug === "acestep-v15-turbo") ||
      state.models.find((model) => model.status.state === "ready") ||
      state.models[0];
    applyModel(preferred);
  }
}

function applyModel(model) {
  state.selectedModel = model;
  setPill(el.modelState, "Locked", "ok");
  el.modelDetails.innerHTML = [
    "<strong>ACE-Step XL Turbo runtime</strong>",
    "Generation uses the active ACE-Step runtime path.",
    "Model selection is locked in this workflow.",
    `Runtime profile shown: ${model.display_name}`,
    `Status: ${model.status.state.replace("_", " ")}`,
  ].join("<br>");
  el.autoInstallModel.checked = false;
  el.autoInstallModel.disabled = true;
  el.installModelButton.disabled = true;
  if (!state.advancedDirty) {
    applyAceDefaults(model);
  }
}

function setNumeric(node, value) {
  node.value = value === null || value === undefined ? "" : String(value);
}

function applyAceDefaults(model) {
  const defaults = model ? { ...(model.generation_defaults || {}), ...(model.repaint_defaults || {}) } : {};
  setNumeric(el.inferenceSteps, defaults.inference_steps);
  setNumeric(el.guidanceScale, defaults.guidance_scale);
  setNumeric(el.shiftValue, defaults.shift);
  setNumeric(el.repaintStrength, defaults.repaint_strength);
  el.repaintMode.value = defaults.repaint_mode || "balanced";
  setNumeric(el.repaintLatentCrossfadeFrames, defaults.repaint_latent_crossfade_frames);
  setNumeric(el.repaintWavCrossfadeSec, defaults.repaint_wav_crossfade_sec);
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
  renderLogList(el.logList, logs);
  renderLogList(el.extractLogList, logs);
  renderLogList(el.musicLogList, logs);
}

function renderLogList(node, logs) {
  node.replaceChildren();
  logs.forEach((entry) => {
    const item = document.createElement("li");
    const level = document.createElement("span");
    level.className = `level ${entry.level}`;
    level.textContent = entry.level;
    const text = document.createTextNode(`${entry.timestamp} ${entry.message}`);
    item.append(level, text);
    node.appendChild(item);
  });
}

function setActivePage(page) {
  el.transitionPage.classList.toggle("active", page === "transition");
  el.extractionPage.classList.toggle("active", page === "extraction");
  el.musicPage.classList.toggle("active", page === "music");
  el.instrumentLabPage.classList.toggle("active", page === "instrument");
  el.audioEditPage.classList.toggle("active", page === "audioedit");
  el.transitionTabButton.classList.toggle("active", page === "transition");
  el.extractionTabButton.classList.toggle("active", page === "extraction");
  el.musicTabButton.classList.toggle("active", page === "music");
  el.instrumentLabTabButton.classList.toggle("active", page === "instrument");
  el.audioEditTabButton.classList.toggle("active", page === "audioedit");
  if (page === "instrument") {
    window.setTimeout(drawInstrumentPianoRoll, 50);
  }
}

function reloadAudioEditor() {
  el.audioEditorFrame.src = "/audiomass/";
}

function openAudioEditorWindow() {
  window.open("/audiomass/", "_blank", "noopener");
}

function assetAudioUrl(asset) {
  return `/api/editor/audio?path=${encodeURIComponent(asset.audio_path)}`;
}

function filteredEditorAssets() {
  const query = el.editorAssetSearch.value.trim().toLowerCase();
  const category = el.editorCategoryFilter.value;
  return state.editorAssets.filter((asset) => {
    if (category !== "all" && asset.category !== category) return false;
    if (!query) return true;
    return [asset.label, asset.category, asset.audio_path, asset.message]
      .join(" ")
      .toLowerCase()
      .includes(query);
  });
}

function renderEditorAssets() {
  el.editorAssetList.replaceChildren();
  const assets = filteredEditorAssets();
  setPill(el.editorAssetState, `${assets.length} shown`, assets.length ? "ok" : "neutral");
  if (!assets.length) {
    const empty = document.createElement("div");
    empty.className = "empty-result";
    empty.textContent = "No matching Dance Station audio assets.";
    el.editorAssetList.appendChild(empty);
    return;
  }

  assets.forEach((asset) => {
    const row = document.createElement("article");
    row.className = "editor-asset-item";
    row.dataset.assetId = asset.asset_id;
    row.innerHTML = `
      <div class="editor-asset-title">
        <strong>${escapeHtml(asset.label)}</strong>
        <span class="category-badge">${escapeHtml(asset.category)}</span>
      </div>
      <audio controls preload="metadata" src="${assetAudioUrl(asset)}"></audio>
      <p class="asset-path">${escapeHtml(asset.audio_path)}</p>
      <div class="rename-row">
        <input class="asset-label-input" type="text" value="${escapeHtml(asset.label)}" aria-label="Asset label" />
        <button class="secondary-button asset-rename-button" type="button">Save Label</button>
      </div>
      <div class="button-row generated-actions">
        <button class="primary-button open-editor-asset-button" type="button">Open in Editor</button>
      </div>
    `;
    row.querySelector(".open-editor-asset-button").addEventListener("click", () => openAssetInEditor(asset));
    row.querySelector(".asset-rename-button").addEventListener("click", () => renameEditorAsset(asset, row));
    el.editorAssetList.appendChild(row);
  });
}

function renderSourceAssetOptions() {
  const selects = [el.sourceAssetSelect, el.extractSourceAssetSelect, el.instrumentAssetSelect].filter(Boolean);
  selects.forEach((select) => {
    const current = select.value;
    select.replaceChildren();
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Choose an existing creation";
    select.appendChild(placeholder);

    state.editorAssets.forEach((asset) => {
      const option = document.createElement("option");
      option.value = asset.asset_id;
      option.textContent = `${asset.category}: ${asset.label}`;
      option.dataset.audioPath = asset.audio_path || "";
      select.appendChild(option);
    });

    if ([...select.options].some((option) => option.value === current)) {
      select.value = current;
    }
  });
}

function selectedSourceAsset(select) {
  const assetId = select.value;
  return state.editorAssets.find((asset) => asset.asset_id === assetId) || null;
}

async function loadExistingCreationAsTransitionSource() {
  const asset = selectedSourceAsset(el.sourceAssetSelect);
  if (!asset || !asset.audio_path) {
    showToast("Choose an existing creation");
    return;
  }
  el.sourcePath.value = asset.audio_path;
  el.selectedFileName.textContent = `${asset.category}: ${asset.label}`;
  await loadSource();
}

async function loadExistingCreationAsExtractionSource() {
  const asset = selectedSourceAsset(el.extractSourceAssetSelect);
  if (!asset || !asset.audio_path) {
    showToast("Choose an existing creation");
    return;
  }
  el.extractSourcePath.value = asset.audio_path;
  el.extractSelectedFileName.textContent = `${asset.category}: ${asset.label}`;
  await loadExtractionSource();
}

function openAssetInEditor(asset) {
  state.selectedEditorAsset = asset;
  const url = assetAudioUrl(asset);
  el.audioEditorFrame.src = `/audiomass/?ds_audio=${encodeURIComponent(url)}&ds_name=${encodeURIComponent(asset.label)}`;
  el.editorCurrentAsset.textContent = `${asset.category}: ${asset.label}`;
  el.editSourceAssetReadout.innerHTML = [
    `<strong>${escapeHtml(asset.label)}</strong>`,
    `Category: ${escapeHtml(asset.category)}`,
    `Path: ${escapeHtml(asset.audio_path)}`,
  ].join("<br>");
  if (!el.editSaveLabelInput.value.trim()) {
    el.editSaveLabelInput.value = `${asset.label} edit`;
  }
  showToast("Loaded asset in Audio Editor");
}

function renameEndpointForAsset(asset) {
  if (asset.category === "transition") return `/api/transitions/${encodeURIComponent(asset.asset_id)}/rename`;
  if (asset.category === "generation") return `/api/music-generations/${encodeURIComponent(asset.asset_id)}/rename`;
  if (asset.category === "edit") return `/api/edits/${encodeURIComponent(asset.asset_id)}/rename`;
  if (asset.category === "instrument" || asset.category === "instrumenttrack") return `/api/instrument-lab/clips/${encodeURIComponent(asset.asset_id)}/rename`;
  if (asset.category === "extraction" || asset.category === "merge") {
    return `/api/extractions/${encodeURIComponent(asset.asset_id)}/rename`;
  }
  return null;
}

async function renameEditorAsset(asset, row) {
  const endpoint = renameEndpointForAsset(asset);
  const input = row.querySelector(".asset-label-input");
  const label = input ? input.value.trim() : "";
  if (!endpoint || !label) {
    showToast("Enter a label");
    return;
  }
  try {
    await api(endpoint, {
      method: "POST",
      body: JSON.stringify({ label }),
    });
    await refreshEditorAssets();
    await refreshExtractions();
    await refreshMusicGenerations();
    await refreshInstrumentClips();
    showToast("Label saved");
  } catch (error) {
    showToast(error.message);
  }
}

async function refreshEditorAssets() {
  state.editorAssets = await api("/api/editor/assets");
  renderSourceAssetOptions();
  renderEditorAssets();
}

function safeEditFileName(label) {
  const stem = label
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80);
  return `${stem || "dance-station-edit"}.wav`;
}

function requestEditorAudio(label) {
  const frameWindow = el.audioEditorFrame && el.audioEditorFrame.contentWindow;
  if (!frameWindow) {
    return Promise.reject(new Error("Audio editor is not ready"));
  }
  const requestId = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      window.removeEventListener("message", onMessage);
      reject(new Error("Audio editor did not return audio"));
    }, 15000);

    function onMessage(event) {
      if (event.source !== frameWindow) return;
      const message = event.data || {};
      if (message.type !== "dance-station-export-audio-result" || message.requestId !== requestId) return;
      window.clearTimeout(timeout);
      window.removeEventListener("message", onMessage);
      if (!message.ok) {
        reject(new Error(message.error || "Audio editor export failed"));
        return;
      }
      const blob = new Blob([message.audio], { type: message.mimeType || "audio/wav" });
      resolve(new File([blob], safeEditFileName(label), { type: "audio/wav" }));
    }

    window.addEventListener("message", onMessage);
    frameWindow.postMessage(
      {
        type: "dance-station-export-audio",
        requestId,
        name: safeEditFileName(label),
      },
      window.location.origin,
    );
  });
}

async function uploadEditedAudioFile(file, label) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("label", label);
  if (state.selectedEditorAsset) {
    formData.append("source_asset_id", state.selectedEditorAsset.asset_id);
    formData.append("source_category", state.selectedEditorAsset.category);
  }

  setPill(el.editSaveState, "Saving", "warn");
  el.saveEditButton.disabled = true;
  try {
    const response = await fetch("/api/edits", {
      method: "POST",
      body: formData,
    });
    const body = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(body && body.detail ? body.detail : `Save failed: ${response.status}`);
    }
    setPill(el.editSaveState, "Saved", "ok");
    el.editSaveFile.value = "";
    el.editSaveFileName.textContent = "No file selected";
    await refreshEditorAssets();
    showToast("Edit saved");
  } catch (error) {
    setPill(el.editSaveState, "Error", "error");
    showToast(error.message);
  } finally {
    el.saveEditButton.disabled = false;
    refreshLogs();
  }
}

async function saveEditedAudio() {
  const fallbackFile = el.editSaveFile.files && el.editSaveFile.files[0];
  const label = el.editSaveLabelInput.value.trim();
  if (!label) {
    showToast("Enter an edit name");
    return;
  }

  setPill(el.editSaveState, "Exporting", "warn");
  el.saveEditButton.disabled = true;
  try {
    const file = await requestEditorAudio(label).catch((error) => {
      if (fallbackFile) return fallbackFile;
      throw error;
    });
    await uploadEditedAudioFile(file, label);
  } catch (error) {
    setPill(el.editSaveState, "Error", "error");
    showToast(error.message);
    el.saveEditButton.disabled = false;
  }
}

function applyMusicModelDefaults() {
  const base = el.musicModelSelect.value === "acestep-v15-base";
  el.musicInferenceSteps.value = base ? "80" : "8";
  el.musicGuidanceScale.value = base ? "0.6" : "1";
  el.musicShift.value = base ? "1" : "3";
  el.musicInferMethod.value = base ? "sde" : "ode";
  el.musicUseTiledDecode.checked = true;
  el.musicDcwEnabled.checked = false;
  el.musicVelocityNormThreshold.value = "0";
  el.musicVelocityEmaFactor.value = "0";
  setPill(el.musicModelState, base ? "Base" : "Turbo", "neutral");
}

function syncMusicVocalControls() {
  const instrumental = el.musicInstrumental.checked;
  el.musicLyrics.disabled = instrumental;
  el.musicVocalLanguage.disabled = instrumental;
  if (instrumental) {
    el.musicLyrics.placeholder = "Instrumental mode sends [Instrumental] to ACE-Step";
  } else {
    el.musicLyrics.placeholder = "Verse and chorus lyrics to sing";
  }
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
      <div class="button-row generated-actions">
        <button class="secondary-button use-source-button" type="button" ${outputPath ? "" : "disabled"}>Use as Source</button>
      </div>
      <dl class="path-list">
        <dt>Message</dt><dd>${result.message}</dd>
        <dt>Mode</dt><dd>${plan.generation_region === "repaint_existing" ? "Repaint existing audio" : "Extend after marker"}</dd>
        <dt>Source</dt><dd>${formatTime(plan.tail_start_seconds)} to ${formatTime(plan.tail_end_seconds)}</dd>
        <dt>Generated</dt><dd>${Number(plan.new_section_seconds || 0).toFixed(1)}s</dd>
        <dt>Repaint before</dt><dd>${Number(plan.repaint_overlap_seconds || 0).toFixed(1)}s</dd>
        <dt>Output</dt><dd>${outputPath || "None"}</dd>
        <dt>Metadata</dt><dd>${result.generated_metadata_path || result.scaffold_metadata_path}</dd>
        <dt>Prompt</dt><dd>${plan.caption}</dd>
      </dl>
    `;
    const useSourceButton = row.querySelector(".use-source-button");
    if (useSourceButton && outputPath) {
      useSourceButton.addEventListener("click", () => useGeneratedAsSource(outputPath));
    }
    el.generatedList.appendChild(row);
  });
}

function renderExtractionTracks() {
  el.extractTrackSelect.replaceChildren();
  state.extractionTracks.forEach((track) => {
    el.extractTrackSelect.appendChild(option(track.replace("_", " "), track));
  });
  if (state.extractionTracks.includes("vocals")) {
    el.extractTrackSelect.value = "vocals";
  }
}

function renderExtractionList() {
  el.extractionList.replaceChildren();
  if (!state.extractionResults.length) {
    const empty = document.createElement("div");
    empty.className = "empty-result";
    empty.textContent = "No extractions yet.";
    el.extractionList.appendChild(empty);
    return;
  }

  state.extractionResults.forEach((item, index) => {
    const row = document.createElement("article");
    row.className = "generated-item";
    row.dataset.extractionId = item.extraction_id || "";
    const outputPath = item.generated_audio_path || "";
    const canMerge = item.type !== "base_test" && item.status === "complete" && outputPath;
    const itemType = item.type === "base_test" ? "Base test" : item.type === "merge" ? "Merge" : "Extraction";
    const sourceLabel = item.type === "base_test" ? "Prompt" : "Source";
    const sourceValue = item.type === "base_test" ? item.prompt || "" : item.source_path || "";
    const displayLabel = item.label || item.track_name || itemType;
    const audio = outputPath
      ? `<audio controls preload="metadata" src="/api/extractions/audio?path=${encodeURIComponent(outputPath)}"></audio>`
      : `<div class="empty-result">No playable audio for this extraction.</div>`;
    const mergeControl = canMerge
      ? `<label class="merge-select"><input class="merge-select-input" type="checkbox" value="${escapeHtml(item.extraction_id)}" /> Select for merge</label>`
      : "";
    const renameControl = item.type !== "base_test"
      ? `
        <div class="rename-row">
          <input class="rename-input" type="text" value="${escapeHtml(displayLabel)}" aria-label="Extraction label" />
          <button class="rename-button secondary-button" type="button">Save Label</button>
        </div>
      `
      : "";
    row.innerHTML = `
      <div class="generated-title">
        <strong>${index === 0 ? "Latest" : itemType} - ${escapeHtml(item.status)}</strong>
        <span>${escapeHtml(displayLabel)}</span>
      </div>
      ${mergeControl}
      ${renameControl}
      ${audio}
      <dl class="path-list">
        <dt>Type</dt><dd>${escapeHtml(itemType)}</dd>
        <dt>Message</dt><dd>${escapeHtml(item.message || "")}</dd>
        <dt>${escapeHtml(sourceLabel)}</dt><dd>${escapeHtml(sourceValue)}</dd>
        <dt>Track</dt><dd>${escapeHtml(item.track_name || "")}</dd>
        <dt>Output</dt><dd>${escapeHtml(outputPath || "None")}</dd>
        <dt>Metadata</dt><dd>${escapeHtml(item.metadata_path || "")}</dd>
      </dl>
    `;
    const renameButton = row.querySelector(".rename-button");
    if (renameButton) {
      renameButton.addEventListener("click", () => renameExtraction(item.extraction_id, row));
    }
    el.extractionList.appendChild(row);
  });
}

function renderMusicList() {
  el.musicList.replaceChildren();
  if (!state.musicResults.length) {
    const empty = document.createElement("div");
    empty.className = "empty-result";
    empty.textContent = "No music generations yet.";
    el.musicList.appendChild(empty);
    return;
  }

  state.musicResults.forEach((item, index) => {
    const row = document.createElement("article");
    row.className = "generated-item";
    const outputPath = item.generated_audio_path || "";
    const audio = outputPath
      ? `<audio controls preload="metadata" src="/api/music-generations/audio?path=${encodeURIComponent(outputPath)}"></audio>`
      : `<div class="empty-result">No playable audio for this generation.</div>`;
    row.innerHTML = `
      <div class="generated-title">
        <strong>${index === 0 ? "Latest" : "Music"} - ${escapeHtml(item.status)}</strong>
        <span>${escapeHtml(item.label || item.model || "music")}</span>
      </div>
      ${audio}
      <dl class="path-list">
        <dt>Message</dt><dd>${escapeHtml(item.message || "")}</dd>
        <dt>Model</dt><dd>${escapeHtml(item.model || "")}</dd>
        <dt>Prompt</dt><dd>${escapeHtml(item.prompt || "")}</dd>
        <dt>Output</dt><dd>${escapeHtml(outputPath || "None")}</dd>
        <dt>Metadata</dt><dd>${escapeHtml(item.metadata_path || "")}</dd>
      </dl>
    `;
    el.musicList.appendChild(row);
  });
}

async function renameExtraction(extractionId, row) {
  const input = row.querySelector(".rename-input");
  const label = input ? input.value.trim() : "";
  if (!label) {
    showToast("Enter a label");
    return;
  }
  try {
    const response = await api(`/api/extractions/${encodeURIComponent(extractionId)}/rename`, {
      method: "POST",
      body: JSON.stringify({ label }),
    });
    const index = state.extractionResults.findIndex((item) => item.extraction_id === extractionId);
    if (index >= 0) state.extractionResults[index] = response.extraction;
    renderExtractionList();
    showToast("Label saved");
  } catch (error) {
    showToast(error.message);
  }
}

async function mergeSelectedExtractions() {
  const ids = Array.from(el.extractionList.querySelectorAll(".merge-select-input:checked")).map((node) => node.value);
  const label = el.mergeLabelInput.value.trim();
  if (ids.length < 2) {
    showToast("Select at least two extraction items");
    return;
  }
  if (!label) {
    showToast("Enter a merge label");
    return;
  }
  el.mergeExtractionsButton.disabled = true;
  el.extractionActivity.innerHTML = "<strong>Merging</strong><br>Combining selected extraction outputs.";
  try {
    const response = await api("/api/extractions/merge", {
      method: "POST",
      body: JSON.stringify({
        extraction_ids: ids,
        label,
        output_format: el.mergeOutputFormat.value,
      }),
    });
    state.extractionResults.unshift(response.extraction);
    state.extractionResults = state.extractionResults.slice(0, 24);
    renderExtractionList();
    await refreshEditorAssets();
    el.extractionActivity.innerHTML = "<strong>Complete</strong><br>Merge finished.";
    showToast("Merge complete");
  } catch (error) {
    el.extractionActivity.innerHTML = `<strong>Error</strong><br>${escapeHtml(error.message)}`;
    showToast(error.message);
  } finally {
    el.mergeExtractionsButton.disabled = false;
    refreshLogs();
  }
}

async function refreshExtractions() {
  state.extractionResults = await api("/api/extractions");
  renderExtractionList();
}

async function refreshMusicGenerations() {
  state.musicResults = await api("/api/music-generations");
  renderMusicList();
}

async function refreshInstrumentClips() {
  state.instrumentClips = await api("/api/instrument-lab/clips");
  renderInstrumentClipList();
}

function activeInstrumentTrack() {
  return state.instrumentTracks.find((track) => track.id === state.activeInstrumentTrackId) || null;
}

function beatDurationSeconds() {
  return 60 / Number(el.instrumentBpm.value || 120);
}

function instrumentTotalBeats() {
  return Math.max(1, Number(el.instrumentBars.value || 4) * 4);
}

function ensureInstrumentLengthForBeat(beat) {
  const neededBars = Math.ceil((beat + 1) / 4);
  const currentBars = Math.max(1, Number(el.instrumentBars.value || 4));
  if (neededBars > currentBars) {
    el.instrumentBars.value = String(neededBars);
  }
}

function instrumentTotalSeconds() {
  return instrumentTotalBeats() * beatDurationSeconds();
}

function midiFrequency(pitch) {
  return 440 * Math.pow(2, (pitch - 69) / 12);
}

function fallbackInstrumentBank() {
  return [
    { id: "synth.lead", name: "Lead Synth", category: "Synths", type: "synth", oscillator: "sawtooth", envelope: { attack: 0.01, release: 0.18 }, octave: 0 },
    { id: "bass.synth", name: "Bass Synth", category: "Bass", type: "synth", oscillator: "square", envelope: { attack: 0.005, release: 0.12 }, octave: -12 },
    { id: "keys.soft-pad", name: "Soft Pad", category: "Keys", type: "synth", oscillator: "triangle", envelope: { attack: 0.14, release: 0.45 }, octave: 0 },
    { id: "keys.pluck", name: "Pluck", category: "Keys", type: "synth", oscillator: "triangle", envelope: { attack: 0.005, release: 0.08 }, octave: 12 },
  ];
}

async function loadInstrumentBank() {
  try {
    const [staticResponse, userResponse] = await Promise.all([
      fetch("/static/instruments/bank.json"),
      fetch("/api/instrument-lab/instruments"),
    ]);
    if (!staticResponse.ok) throw new Error(`Instrument bank unavailable: ${staticResponse.status}`);
    const body = await staticResponse.json();
    const userInstruments = userResponse.ok ? await userResponse.json() : [];
    state.instrumentBank = [
      ...(Array.isArray(body.instruments) ? body.instruments : fallbackInstrumentBank()),
      ...(Array.isArray(userInstruments) ? userInstruments : []),
    ];
    setPill(el.instrumentBankState, `${state.instrumentBank.length} loaded`, "ok");
  } catch (error) {
    state.instrumentBank = fallbackInstrumentBank();
    setPill(el.instrumentBankState, "Fallback", "warn");
  }
  renderInstrumentBankOptions();
}

function instrumentDefinition(instrumentId) {
  return state.instrumentBank.find((instrument) => instrument.id === instrumentId)
    || state.instrumentBank[0]
    || fallbackInstrumentBank()[0];
}

function legacyInstrumentId(instrumentId) {
  const aliases = {
    lead: "synth.lead",
    bass: "bass.synth",
    pad: "keys.soft-pad",
    pluck: "keys.pluck",
  };
  return aliases[instrumentId] || instrumentId || "synth.lead";
}

function renderInstrumentBankOptions() {
  const current = legacyInstrumentId(el.instrumentPatch.value || activeInstrumentTrack()?.instrument || "synth.lead");
  el.instrumentPatch.replaceChildren();
  const groups = new Map();
  state.instrumentBank.forEach((instrument) => {
    const category = instrument.category || "Other";
    if (!groups.has(category)) groups.set(category, []);
    groups.get(category).push(instrument);
  });
  groups.forEach((instruments, category) => {
    const group = document.createElement("optgroup");
    group.label = category;
    instruments.forEach((instrument) => {
      const option = document.createElement("option");
      option.value = instrument.id;
      option.textContent = instrument.name;
      group.appendChild(option);
    });
    el.instrumentPatch.appendChild(group);
  });
  if ([...el.instrumentPatch.options].some((option) => option.value === current)) {
    el.instrumentPatch.value = current;
  }
  updateInstrumentInfo();
}

function updateInstrumentInfo() {
  const instrument = instrumentDefinition(el.instrumentPatch.value || activeInstrumentTrack()?.instrument);
  const sampleCount = Array.isArray(instrument.samples) ? `; ${instrument.samples.length} sample${instrument.samples.length === 1 ? "" : "s"}` : "";
  el.instrumentInfo.innerHTML = `<strong>${escapeHtml(instrument.name)}</strong><br>${escapeHtml(instrument.category || "Other")} / ${escapeHtml(instrument.type || "synth")}${escapeHtml(sampleCount)}`;
}

async function importSfzInstrument() {
  const sfzFile = el.sfzInstrumentFile.files && el.sfzInstrumentFile.files[0];
  const label = el.sfzInstrumentLabel.value.trim() || (sfzFile ? sfzFile.name.replace(/\.sfz$/i, "") : "");
  if (!sfzFile) {
    showToast("Choose an SFZ file");
    return;
  }
  if (!label) {
    showToast("Enter an instrument name");
    return;
  }
  el.importSfzButton.disabled = true;
  setPill(el.instrumentBankState, "Importing SFZ", "warn");
  try {
    const formData = new FormData();
    formData.append("label", label);
    formData.append("sfz_file", sfzFile);
    Array.from(el.sfzSampleFiles.files || []).forEach((file) => {
      formData.append("sample_files", file);
    });
    const response = await fetch("/api/instrument-lab/instruments/sfz", { method: "POST", body: formData });
    const body = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(body && body.detail ? body.detail : `SFZ import failed: ${response.status}`);
    }
    await loadInstrumentBank();
    const instrumentId = body.instrument && body.instrument.id;
    if (instrumentId && [...el.instrumentPatch.options].some((option) => option.value === instrumentId)) {
      el.instrumentPatch.value = instrumentId;
      const active = activeInstrumentTrack();
      if (active && active.kind === "instrument") active.instrument = instrumentId;
    }
    updateInstrumentInfo();
    el.sfzInstrumentFile.value = "";
    el.sfzSampleFiles.value = "";
    showToast("SFZ instrument imported");
  } catch (error) {
    setPill(el.instrumentBankState, "Import failed", "bad");
    el.instrumentInfo.innerHTML = `<strong>SFZ import failed</strong><br>${escapeHtml(error.message)}`;
    showToast(error.message);
  } finally {
    el.importSfzButton.disabled = false;
  }
}

function ensureInstrumentAudioContext() {
  if (!state.instrumentAudioContext) {
    state.instrumentAudioContext = new AudioContext();
  }
  return state.instrumentAudioContext;
}

function scheduleSynthNote(context, destination, note, track, offsetSeconds = 0) {
  const instrument = instrumentDefinition(legacyInstrumentId(track.instrument || el.instrumentPatch.value));
  if (instrument.type !== "synth") {
    return scheduleSampleNote(context, destination, note, track, instrument, offsetSeconds);
  }
  const envelope = instrument.envelope || {};
  const start = offsetSeconds + note.start * beatDurationSeconds();
  const duration = Math.max(0.05, note.duration * beatDurationSeconds());
  const gain = context.createGain();
  const oscillator = context.createOscillator();
  const volume = Number(track.volume ?? 0.85) * Number(el.instrumentMasterVolume.value || 0.8) * Number(note.velocity ?? 0.85);
  oscillator.type = instrument.oscillator || "sine";
  oscillator.frequency.setValueAtTime(midiFrequency(note.pitch + Number(instrument.octave || 0)), start);
  gain.gain.setValueAtTime(0, start);
  gain.gain.linearRampToValueAtTime(volume, start + Number(envelope.attack ?? 0.01));
  gain.gain.setValueAtTime(volume, start + Math.max(Number(envelope.attack ?? 0.01), duration - Number(envelope.release ?? 0.18)));
  gain.gain.linearRampToValueAtTime(0, start + duration + Number(envelope.release ?? 0.18));
  oscillator.connect(gain).connect(destination);
  oscillator.start(start);
  oscillator.stop(start + duration + Number(envelope.release ?? 0.18) + 0.05);
  return oscillator;
}

function scheduleSampleNote(context, destination, note, track, instrument, offsetSeconds = 0) {
  const region = sampleRegionForNote(instrument, note.pitch);
  const sampleBuffer = region ? state.instrumentSampleBufferCache.get(sampleRegionCacheKey(region)) : null;
  if (!region || !sampleBuffer) {
    return scheduleFallbackSampleNote(context, destination, note, track, instrument, offsetSeconds);
  }
  const start = offsetSeconds + note.start * beatDurationSeconds();
  const duration = Math.max(0.05, note.duration * beatDurationSeconds());
  const source = context.createBufferSource();
  const gain = context.createGain();
  const volume = Number(track.volume ?? 0.85) * Number(el.instrumentMasterVolume.value || 0.8) * Number(note.velocity ?? 0.85);
  const attack = Number(instrument.envelope?.attack ?? 0.005);
  const release = Number(instrument.envelope?.release ?? 0.2);
  source.buffer = sampleBuffer;
  source.playbackRate.setValueAtTime(Math.pow(2, (note.pitch - Number(region.root ?? region.note ?? note.pitch)) / 12), start);
  gain.gain.setValueAtTime(0, start);
  gain.gain.linearRampToValueAtTime(volume, start + attack);
  gain.gain.setValueAtTime(volume, start + Math.max(attack, duration - release));
  gain.gain.linearRampToValueAtTime(0, start + duration + release);
  source.connect(gain).connect(destination);
  source.start(start);
  source.stop(start + duration + release + 0.05);
  return source;
}

function scheduleFallbackSampleNote(context, destination, note, track, instrument, offsetSeconds = 0) {
  const start = offsetSeconds + note.start * beatDurationSeconds();
  const duration = Math.max(0.05, note.duration * beatDurationSeconds());
  const gain = context.createGain();
  const volume = Number(track.volume ?? 0.85) * Number(el.instrumentMasterVolume.value || 0.8) * Number(note.velocity ?? 0.85);
  const release = Number(instrument.envelope?.release ?? 0.2);
  gain.gain.setValueAtTime(volume, start);
  gain.gain.linearRampToValueAtTime(0, start + duration + release);
  gain.connect(destination);
  const oscillator = context.createOscillator();
  oscillator.type = "sine";
  oscillator.frequency.setValueAtTime(midiFrequency(note.pitch), start);
  oscillator.connect(gain);
  oscillator.start(start);
  oscillator.stop(start + duration + release);
  return oscillator;
}

function sampleRegionForNote(instrument, pitch) {
  const regions = Array.isArray(instrument.samples) ? instrument.samples : [];
  if (!regions.length) return null;
  const matching = regions.filter((region) => pitch >= Number(region.low ?? region.note ?? 0) && pitch <= Number(region.high ?? region.note ?? 127));
  const candidates = matching.length ? matching : regions;
  return candidates
    .map((region) => ({ region, distance: Math.abs(pitch - Number(region.root ?? region.note ?? pitch)) }))
    .sort((left, right) => left.distance - right.distance)[0].region;
}

function sampleRegionCacheKey(region) {
  return region.url || region.path || "";
}

function sampleRegionUrl(region) {
  if (region.url) return region.url;
  return `/static/instruments/${region.path}`;
}

async function loadInstrumentSample(context, region) {
  const cacheKey = sampleRegionCacheKey(region);
  if (state.instrumentSampleBufferCache.has(cacheKey)) {
    return state.instrumentSampleBufferCache.get(cacheKey);
  }
  const response = await fetch(sampleRegionUrl(region));
  if (!response.ok) {
    throw new Error(`Could not load instrument sample: ${region.path || region.url}`);
  }
  const arrayBuffer = await response.arrayBuffer();
  const buffer = await context.decodeAudioData(arrayBuffer.slice(0));
  state.instrumentSampleBufferCache.set(cacheKey, buffer);
  return buffer;
}

async function prepareInstrumentSamples(context, tracks) {
  const paths = new Set();
  tracks.forEach((track) => {
    if (track.kind !== "instrument") return;
    const instrument = instrumentDefinition(legacyInstrumentId(track.instrument || el.instrumentPatch.value));
    if (instrument.type !== "sample") return;
    (instrument.samples || []).forEach((sample) => {
      if (sample.path || sample.url) paths.add(sample);
    });
  });
  for (const sample of paths) {
    setPill(el.instrumentBankState, "Loading samples", "warn");
    await loadInstrumentSample(context, sample);
  }
  if (paths.size) {
    setPill(el.instrumentBankState, `${state.instrumentBank.length} loaded`, "ok");
  }
}

async function decodedAssetBuffer(context, audioPath) {
  if (state.instrumentAudioBufferCache.has(audioPath)) {
    return state.instrumentAudioBufferCache.get(audioPath);
  }
  const response = await fetch(`/api/editor/audio?path=${encodeURIComponent(audioPath)}`);
  if (!response.ok) {
    throw new Error(`Could not load audio track: ${response.status}`);
  }
  const arrayBuffer = await response.arrayBuffer();
  const buffer = await context.decodeAudioData(arrayBuffer.slice(0));
  state.instrumentAudioBufferCache.set(audioPath, buffer);
  return buffer;
}

function stopInstrumentSources() {
  if (state.instrumentCountdownTimer) {
    window.clearInterval(state.instrumentCountdownTimer);
    state.instrumentCountdownTimer = null;
  }
  state.instrumentPlayingSources.forEach((source) => {
    try {
      source.stop();
    } catch (error) {
      // Source may already be stopped.
    }
  });
  state.instrumentPlayingSources = [];
  state.instrumentTransportStartTime = null;
  state.instrumentTransportStartBeat = 0;
}

function setInstrumentRecording(enabled) {
  state.instrumentRecording = enabled;
  el.recordInstrumentButton.classList.toggle("active", enabled);
}

async function prepareInstrumentAudioTracks(context, playbackId) {
  const tracks = state.instrumentTracks.filter((track) => shouldTrackPlayInCurrentPass(track) && track.kind === "audio" && track.audio_path);
  const buffers = new Map();
  for (let index = 0; index < tracks.length; index += 1) {
    if (playbackId !== state.instrumentPlaybackId) return null;
    setPill(el.instrumentLabState, `Preparing ${index + 1}/${tracks.length}`, "warn");
    const buffer = await decodedAssetBuffer(context, tracks[index].audio_path);
    if (playbackId !== state.instrumentPlaybackId) return null;
    buffers.set(tracks[index].id, buffer);
  }
  return buffers;
}

function shouldTrackPlayInCurrentPass(track) {
  if (track.muted) return false;
  if (state.instrumentRecording && track.playDuringRecord === false) return false;
  return true;
}

function updateInstrumentTransportStatus(message, kind = "ok") {
  const suffix = state.instrumentRecording ? " + Recording" : "";
  setPill(el.instrumentLabState, `${message}${suffix}`, kind);
}

function transportBeatAtCurrentTime() {
  if (!state.instrumentAudioContext || state.instrumentTransportStartTime === null) return null;
  const elapsed = state.instrumentAudioContext.currentTime - state.instrumentTransportStartTime;
  return state.instrumentTransportStartBeat + elapsed / beatDurationSeconds();
}

function setInstrumentCursorBeat(beat) {
  const bounds = pianoRollContentBounds();
  state.instrumentCursorBeat = Math.max(0, Math.min(bounds.totalBeats, quantizeBeat(beat)));
  drawInstrumentPianoRoll();
}

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

async function runInstrumentCountIn(playbackId, seconds = 2) {
  let remaining = seconds;
  setPill(el.instrumentLabState, `Recording starts in ${remaining}`, "warn");
  state.instrumentCountdownTimer = window.setInterval(() => {
    remaining -= 1;
    if (remaining > 0 && playbackId === state.instrumentPlaybackId) {
      setPill(el.instrumentLabState, `Recording starts in ${remaining}`, "warn");
    }
  }, 1000);
  await sleep(seconds * 1000);
  if (state.instrumentCountdownTimer) {
    window.clearInterval(state.instrumentCountdownTimer);
    state.instrumentCountdownTimer = null;
  }
}

async function startInstrumentTransport({ recording = false } = {}) {
  state.instrumentPlaybackId += 1;
  const playbackId = state.instrumentPlaybackId;
  stopInstrumentSources();
  setInstrumentRecording(recording);
  const context = ensureInstrumentAudioContext();
  el.playInstrumentButton.disabled = true;
  el.recordInstrumentButton.disabled = true;
  setPill(el.instrumentLabState, recording ? "Preparing recording" : "Preparing playback", "warn");

  try {
    await context.resume();
    if (playbackId !== state.instrumentPlaybackId) return;

    const audioBuffers = await prepareInstrumentAudioTracks(context, playbackId);
    if (playbackId !== state.instrumentPlaybackId || !audioBuffers) return;
    await prepareInstrumentSamples(context, state.instrumentTracks.filter((track) => shouldTrackPlayInCurrentPass(track)));
    if (playbackId !== state.instrumentPlaybackId) return;

    if (recording) {
      await runInstrumentCountIn(playbackId, 2);
      if (playbackId !== state.instrumentPlaybackId) return;
    }

    const destination = context.destination;
    const startAt = context.currentTime + 0.05;
    const cursorBeat = Math.max(0, Math.min(pianoRollContentBounds().totalBeats, state.instrumentCursorBeat || 0));
    const cursorSeconds = cursorBeat * beatDurationSeconds();
    state.instrumentTransportStartTime = startAt;
    state.instrumentTransportStartBeat = cursorBeat;
    for (const track of state.instrumentTracks) {
      if (!shouldTrackPlayInCurrentPass(track)) continue;
      if (track.kind === "audio" && track.audio_path) {
        const buffer = audioBuffers.get(track.id);
        if (!buffer) continue;
        const source = context.createBufferSource();
        const gain = context.createGain();
        gain.gain.value = Number(track.volume ?? 0.85) * Number(el.instrumentMasterVolume.value || 0.8);
        source.buffer = buffer;
        source.connect(gain).connect(destination);
        if (cursorSeconds >= buffer.duration) continue;
        source.start(startAt, cursorSeconds);
        state.instrumentPlayingSources.push(source);
        continue;
      }
      for (const note of track.notes || []) {
        if (note.start + note.duration < cursorBeat) continue;
        const trimBeats = Math.max(0, cursorBeat - note.start);
        const scheduledNote = {
          ...note,
          start: Math.max(0, note.start - cursorBeat),
          duration: Math.max(0.05, note.duration - trimBeats),
        };
        state.instrumentPlayingSources.push(scheduleSynthNote(context, destination, scheduledNote, track, startAt));
      }
    }
    updateInstrumentTransportStatus(recording ? "Recording" : "Playing", recording ? "warn" : "ok");
  } catch (error) {
    if (playbackId === state.instrumentPlaybackId) {
      setPill(el.instrumentLabState, "Playback failed", "bad");
      showToast(error.message || "Playback failed.");
    }
  } finally {
    if (playbackId === state.instrumentPlaybackId) {
      el.playInstrumentButton.disabled = false;
      el.recordInstrumentButton.disabled = false;
    }
  }
}

async function playInstrumentLab() {
  await startInstrumentTransport({ recording: false });
}

async function recordInstrumentLab() {
  await startInstrumentTransport({ recording: true });
}

function stopInstrumentLab() {
  state.instrumentPlaybackId += 1;
  stopInstrumentSources();
  setInstrumentRecording(false);
  el.playInstrumentButton.disabled = false;
  el.recordInstrumentButton.disabled = false;
  setPill(el.instrumentLabState, "Ready", "neutral");
}

function audioBufferToWav(buffer) {
  const channels = buffer.numberOfChannels;
  const sampleRate = buffer.sampleRate;
  const samples = buffer.length;
  const bytesPerSample = 2;
  const blockAlign = channels * bytesPerSample;
  const dataSize = samples * blockAlign;
  const wav = new ArrayBuffer(44 + dataSize);
  const view = new DataView(wav);
  const writeString = (offset, value) => {
    for (let i = 0; i < value.length; i += 1) view.setUint8(offset + i, value.charCodeAt(i));
  };
  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, dataSize, true);
  let offset = 44;
  for (let i = 0; i < samples; i += 1) {
    for (let channel = 0; channel < channels; channel += 1) {
      let value = buffer.getChannelData(channel)[i];
      value = Math.max(-1, Math.min(1, value));
      view.setInt16(offset, value < 0 ? value * 0x8000 : value * 0x7fff, true);
      offset += bytesPerSample;
    }
  }
  return wav;
}

async function renderInstrumentLabBuffer(tracks = state.instrumentTracks) {
  const duration = instrumentTotalSeconds();
  const context = new OfflineAudioContext(2, Math.ceil(44100 * duration), 44100);
  const master = context.createGain();
  master.gain.value = Number(el.instrumentMasterVolume.value || 0.8);
  master.connect(context.destination);
  await prepareInstrumentSamples(context, tracks);

  for (const track of tracks) {
    if (track.muted) continue;
    if (track.kind === "audio" && track.audio_path) {
      const buffer = await decodedAssetBuffer(context, track.audio_path);
      const source = context.createBufferSource();
      const gain = context.createGain();
      gain.gain.value = Number(track.volume ?? 0.85);
      source.buffer = buffer;
      source.connect(gain).connect(master);
      source.start(0);
      continue;
    }
    for (const note of track.notes || []) {
      scheduleSynthNote(context, master, note, track, 0);
    }
  }
  return context.startRendering();
}

function instrumentProjectPayload(tracks = state.instrumentTracks, activeTrackId = state.activeInstrumentTrackId) {
  return {
    bpm: Number(el.instrumentBpm.value || 120),
    key: el.instrumentKey.value.trim(),
    bars: Number(el.instrumentBars.value || 4),
    master_volume: Number(el.instrumentMasterVolume.value || 0.8),
    active_track_id: activeTrackId,
    cursor_beat: state.instrumentCursorBeat || 0,
    tracks,
  };
}

async function renderInstrumentPreview(tracks = state.instrumentTracks, label = el.instrumentClipLabel.value || "instrument") {
  setPill(el.instrumentRenderState, "Rendering", "warn");
  try {
    const buffer = await renderInstrumentLabBuffer(tracks);
    const wav = audioBufferToWav(buffer);
    if (state.instrumentPreviewUrl) URL.revokeObjectURL(state.instrumentPreviewUrl);
    state.instrumentPreviewUrl = URL.createObjectURL(new Blob([wav], { type: "audio/wav" }));
    el.instrumentPreviewAudio.src = state.instrumentPreviewUrl;
    setPill(el.instrumentRenderState, "Rendered", "ok");
    return new File([wav], `${safeEditFileName(label)}`, { type: "audio/wav" });
  } catch (error) {
    setPill(el.instrumentRenderState, "Error", "error");
    showToast(error.message);
    throw error;
  }
}

async function saveInstrumentClip({ trackOnly = false } = {}) {
  const active = activeInstrumentTrack();
  if (trackOnly && (!active || active.kind !== "instrument")) {
    showToast("Select an instrument track to save");
    return;
  }
  const defaultLabel = trackOnly && active ? active.label : el.instrumentClipLabel.value;
  const label = (defaultLabel || "").trim();
  if (!label) {
    showToast("Enter a clip name");
    return;
  }
  const saveButton = trackOnly ? el.saveInstrumentTrackButton : el.saveInstrumentButton;
  saveButton.disabled = true;
  try {
    const tracks = trackOnly ? [{ ...active, muted: false }] : state.instrumentTracks;
    const file = await renderInstrumentPreview(tracks, label);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("label", label);
    formData.append("clip_type", trackOnly ? "instrumenttrack" : "instrument");
    formData.append("project_json", JSON.stringify(instrumentProjectPayload(tracks, trackOnly ? active.id : state.activeInstrumentTrackId)));
    const response = await fetch("/api/instrument-lab/clips", { method: "POST", body: formData });
    const body = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(body && body.detail ? body.detail : `Save failed: ${response.status}`);
    }
    await refreshInstrumentClips();
    await refreshEditorAssets();
    showToast(trackOnly ? "Instrument track saved" : "Instrument clip saved");
  } catch (error) {
    showToast(error.message);
  } finally {
    saveButton.disabled = false;
  }
}

function renderInstrumentTracks() {
  el.instrumentTrackList.replaceChildren();
  for (const track of state.instrumentTracks) {
    const row = document.createElement("article");
    row.className = `instrument-track-item${track.id === state.activeInstrumentTrackId ? " active" : ""}`;
    row.innerHTML = `
      <button class="track-select-button" type="button">${escapeHtml(track.label)}</button>
      <span class="category-badge">${escapeHtml(track.kind)}</span>
      <input class="track-volume" type="number" min="0" max="1" step="0.05" value="${Number(track.volume ?? 0.85)}" aria-label="Track volume" />
      <label class="track-toggle"><input type="checkbox" ${track.playDuringRecord === false ? "" : "checked"} /> Play in record</label>
      <label class="track-mute"><input type="checkbox" ${track.muted ? "checked" : ""} /> Mute</label>
    `;
    row.querySelector(".track-select-button").addEventListener("click", () => {
      state.activeInstrumentTrackId = track.id;
      setSelectedInstrumentNotes([]);
      renderInstrumentTracks();
      drawInstrumentPianoRoll();
    });
    row.querySelector(".track-volume").addEventListener("change", (event) => {
      track.volume = Number(event.target.value || 0.85);
    });
    row.querySelector(".track-toggle input").addEventListener("change", (event) => {
      track.playDuringRecord = event.target.checked;
    });
    row.querySelector(".track-mute input").addEventListener("change", (event) => {
      track.muted = event.target.checked;
      renderInstrumentTracks();
    });
    el.instrumentTrackList.appendChild(row);
  }
  const active = activeInstrumentTrack();
  el.instrumentActiveTrackReadout.textContent = active ? `${active.kind}: ${active.label}` : "No track selected";
  if (active && active.kind === "instrument") {
    const instrumentId = legacyInstrumentId(active.instrument);
    if ([...el.instrumentPatch.options].some((option) => option.value === instrumentId)) {
      el.instrumentPatch.value = instrumentId;
    }
  }
  updateInstrumentInfo();
}

function addInstrumentTrack() {
  const count = state.instrumentTracks.filter((track) => track.kind === "instrument").length + 1;
  const track = {
    id: `track-${Date.now().toString(16)}`,
    label: `Track ${count}`,
    kind: "instrument",
    instrument: legacyInstrumentId(el.instrumentPatch.value || "synth.lead"),
    volume: 0.85,
    pan: 0,
    muted: false,
    playDuringRecord: true,
    notes: [],
  };
  state.instrumentTracks.push(track);
  state.activeInstrumentTrackId = track.id;
  renderInstrumentTracks();
  drawInstrumentPianoRoll();
}

function importInstrumentAssetTrack() {
  const asset = selectedSourceAsset(el.instrumentAssetSelect);
  if (!asset || !asset.audio_path) {
    showToast("Choose an existing creation");
    return;
  }
  const track = {
    id: `audio-${Date.now().toString(16)}`,
    label: asset.label,
    kind: "audio",
    source_asset_id: asset.asset_id,
    category: asset.category,
    audio_path: asset.audio_path,
    duration_seconds: Number(asset.duration_seconds || 0),
    volume: 0.85,
    pan: 0,
    muted: false,
    playDuringRecord: true,
    notes: [],
  };
  state.instrumentTracks.push(track);
  state.activeInstrumentTrackId = track.id;
  renderInstrumentTracks();
  drawInstrumentPianoRoll();
  showToast("Audio track added");
}

function renderInstrumentClipList() {
  el.instrumentClipList.replaceChildren();
  if (!state.instrumentClips.length) {
    const empty = document.createElement("div");
    empty.className = "empty-result";
    empty.textContent = "No instrument clips yet.";
    el.instrumentClipList.appendChild(empty);
    return;
  }
  state.instrumentClips.forEach((clip) => {
    const item = document.createElement("article");
    item.className = "generated-item";
    item.innerHTML = `
      <div class="generated-title">
        <strong>${escapeHtml(clip.label || clip.clip_id)}</strong>
        <span>${escapeHtml(clip.type || "instrument")}</span>
      </div>
      <div class="button-row generated-actions">
        <button class="secondary-button load-instrument-project-button" type="button">Load</button>
      </div>
      <audio controls preload="metadata" src="/api/instrument-lab/audio?path=${encodeURIComponent(clip.generated_audio_path)}"></audio>
      <p>${escapeHtml(clip.message || "")}</p>
    `;
    item.querySelector(".load-instrument-project-button").addEventListener("click", () => loadInstrumentProject(clip));
    el.instrumentClipList.appendChild(item);
  });
}

function loadInstrumentProject(clip) {
  const project = clip.project || {};
  if (!Array.isArray(project.tracks) || !project.tracks.length) {
    showToast("Saved clip has no editable project data");
    return;
  }
  state.instrumentTracks = project.tracks.map((track, index) => ({
    id: track.id || `track-${Date.now().toString(16)}-${index}`,
    label: track.label || `Track ${index + 1}`,
    kind: track.kind || "instrument",
    instrument: legacyInstrumentId(track.instrument || "synth.lead"),
    volume: Number(track.volume ?? 0.85),
    pan: Number(track.pan ?? 0),
    muted: Boolean(track.muted),
    playDuringRecord: track.playDuringRecord !== false,
    source_asset_id: track.source_asset_id || "",
    category: track.category || "",
    audio_path: track.audio_path || "",
    duration_seconds: Number(track.duration_seconds || 0),
    notes: Array.isArray(track.notes) ? track.notes : [],
  }));
  state.activeInstrumentTrackId = project.active_track_id || state.instrumentTracks[0].id;
  state.instrumentCursorBeat = Number(project.cursor_beat || 0);
  el.instrumentBpm.value = String(project.bpm || 120);
  el.instrumentKey.value = project.key || "";
  el.instrumentBars.value = String(project.bars || 4);
  el.instrumentMasterVolume.value = String(project.master_volume ?? 0.8);
  setSelectedInstrumentNotes([]);
  renderInstrumentTracks();
  drawInstrumentPianoRoll();
  showToast("Instrument project loaded");
}

const PIANO_MIN_PITCH = 21;
const PIANO_MAX_PITCH = 108;
const PIANO_ROLL_RULER_HEIGHT = 28;

function allInstrumentNotes() {
  return state.instrumentTracks.flatMap((track) => track.kind === "instrument" ? (track.notes || []) : []);
}

function pianoRollContentBounds() {
  const notes = allInstrumentNotes();
  const noteEnd = notes.reduce((maximum, note) => Math.max(maximum, Number(note.start || 0) + Number(note.duration || 0)), 0);
  const audioEnd = state.instrumentTracks.reduce((maximum, track) => {
    if (track.kind !== "audio") return maximum;
    const durationSeconds = Number(track.duration_seconds || 0);
    return Math.max(maximum, durationSeconds / beatDurationSeconds());
  }, 0);
  const noteMinPitch = notes.reduce((minimum, note) => Math.min(minimum, Number(note.pitch || 60)), PIANO_MAX_PITCH);
  const noteMaxPitch = notes.reduce((maximum, note) => Math.max(maximum, Number(note.pitch || 60)), PIANO_MIN_PITCH);
  const minPitch = notes.length ? Math.max(0, Math.min(PIANO_MIN_PITCH, noteMinPitch - 4)) : PIANO_MIN_PITCH;
  const maxPitch = notes.length ? Math.min(127, Math.max(PIANO_MAX_PITCH, noteMaxPitch + 4)) : PIANO_MAX_PITCH;
  return {
    totalBeats: Math.max(instrumentTotalBeats(), Math.ceil(Math.max(noteEnd, audioEnd) + 1)),
    minPitch,
    maxPitch,
    noteStart: notes.length ? Math.max(0, Math.min(...notes.map((note) => Number(note.start || 0)))) : 0,
    noteEnd,
    noteMinPitch: notes.length ? noteMinPitch : minPitch,
    noteMaxPitch: notes.length ? noteMaxPitch : maxPitch,
  };
}

function clampPianoRollView() {
  const bounds = pianoRollContentBounds();
  const totalBeats = bounds.totalBeats;
  const totalPitches = bounds.maxPitch - bounds.minPitch + 1;
  const view = state.pianoRollView;
  view.visibleBeats = Math.max(1, Math.min(totalBeats, view.visibleBeats || totalBeats));
  view.visiblePitches = Math.max(12, Math.min(totalPitches, view.visiblePitches || totalPitches));
  view.beatOffset = Math.max(0, Math.min(totalBeats - view.visibleBeats, view.beatOffset || 0));
  view.pitchOffset = Math.max(bounds.minPitch, Math.min(bounds.maxPitch - view.visiblePitches + 1, view.pitchOffset || bounds.minPitch));
}

function updatePianoRollViewportReadout() {
  if (!el.pianoRollViewportReadout) return;
  const view = state.pianoRollView;
  const bounds = pianoRollContentBounds();
  const beatStart = view.beatOffset;
  const beatEnd = view.beatOffset + view.visibleBeats;
  const pitchEnd = view.pitchOffset + view.visiblePitches - 1;
  el.pianoRollViewportReadout.textContent = `Beats ${beatStart.toFixed(1)}-${beatEnd.toFixed(1)} | MIDI ${view.pitchOffset}-${pitchEnd}`;
  if (el.pianoRollScroll) {
    const max = Math.max(0, bounds.totalBeats - view.visibleBeats);
    el.pianoRollScroll.max = String(max);
    el.pianoRollScroll.value = String(Math.min(max, view.beatOffset));
    el.pianoRollScroll.disabled = max <= 0;
  }
}

function pianoRollMetrics() {
  clampPianoRollView();
  const canvas = el.instrumentPianoRoll;
  const rect = canvas.getBoundingClientRect();
  const width = canvas.width;
  const height = canvas.height;
  const noteHeight = Math.max(1, height - PIANO_ROLL_RULER_HEIGHT);
  const view = state.pianoRollView;
  return {
    canvas,
    width,
    height,
    rulerHeight: PIANO_ROLL_RULER_HEIGHT,
    noteHeight,
    beatWidth: width / view.visibleBeats,
    pitchHeight: noteHeight / view.visiblePitches,
    scaleX: width / rect.width,
    scaleY: height / rect.height,
    beatOffset: view.beatOffset,
    pitchOffset: view.pitchOffset,
    visibleBeats: view.visibleBeats,
    visiblePitches: view.visiblePitches,
  };
}

function drawInstrumentPianoRoll() {
  const canvas = el.instrumentPianoRoll;
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const metrics = pianoRollMetrics();
  const active = activeInstrumentTrack();
  ctx.clearRect(0, 0, metrics.width, metrics.height);
  ctx.fillStyle = "#111317";
  ctx.fillRect(0, 0, metrics.width, metrics.height);
  ctx.fillStyle = "#181b21";
  ctx.fillRect(0, 0, metrics.width, metrics.rulerHeight);

  const topPitch = metrics.pitchOffset + metrics.visiblePitches - 1;
  for (let pitch = metrics.pitchOffset; pitch <= topPitch; pitch += 1) {
    const y = pitchToCanvasY(pitch, metrics);
    const isBlack = [1, 3, 6, 8, 10].includes(pitch % 12);
    ctx.fillStyle = isBlack ? "rgba(255,255,255,0.025)" : "rgba(255,255,255,0.045)";
    ctx.fillRect(0, y, metrics.width, metrics.pitchHeight);
  }
  const firstBeat = Math.floor(metrics.beatOffset);
  const lastBeat = Math.ceil(metrics.beatOffset + metrics.visibleBeats);
  for (let beat = firstBeat; beat <= lastBeat; beat += 1) {
    const x = beatToCanvasX(beat, metrics);
    ctx.strokeStyle = beat % 4 === 0 ? "rgba(242,184,75,0.45)" : "rgba(255,255,255,0.12)";
    ctx.beginPath();
    ctx.moveTo(x, metrics.rulerHeight);
    ctx.lineTo(x, metrics.height);
    ctx.stroke();
    if (beat % 4 === 0) {
      ctx.fillStyle = "rgba(231,235,242,0.72)";
      ctx.font = "12px system-ui, sans-serif";
      ctx.fillText(String(beat), x + 4, 18);
    }
  }
  drawPianoRollSelection(ctx, metrics);
  drawPianoRollCursor(ctx, metrics);
  updatePianoRollViewportReadout();
  if (!active || active.kind !== "instrument") return;

  active.notes.forEach((note) => {
    if (!noteIntersectsView(note, metrics)) return;
    const x = beatToCanvasX(note.start, metrics);
    const y = pitchToCanvasY(note.pitch, metrics);
    const w = Math.max(6, note.duration * metrics.beatWidth);
    const h = Math.max(8, metrics.pitchHeight - 2);
    ctx.fillStyle = state.selectedInstrumentNoteIds.includes(note.id) || note.id === state.selectedInstrumentNoteId ? "#f2b84b" : "#2dd4bf";
    ctx.fillRect(x, y + 1, w, h);
    ctx.strokeStyle = "rgba(0,0,0,0.45)";
    ctx.strokeRect(x, y + 1, w, h);
  });
}

function beatToCanvasX(beat, metrics) {
  return (beat - metrics.beatOffset) * metrics.beatWidth;
}

function pitchToCanvasY(pitch, metrics) {
  return metrics.rulerHeight + metrics.noteHeight - (pitch - metrics.pitchOffset + 1) * metrics.pitchHeight;
}

function canvasXToBeat(x, metrics) {
  return metrics.beatOffset + x / metrics.beatWidth;
}

function canvasYToPitch(y, metrics) {
  const bounds = pianoRollContentBounds();
  return Math.max(bounds.minPitch, Math.min(bounds.maxPitch, metrics.pitchOffset + Math.floor((metrics.rulerHeight + metrics.noteHeight - y) / metrics.pitchHeight)));
}

function drawPianoRollCursor(ctx, metrics) {
  const x = beatToCanvasX(state.instrumentCursorBeat || 0, metrics);
  if (x < 0 || x > metrics.width) return;
  ctx.strokeStyle = "#f2b84b";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x, 0);
  ctx.lineTo(x, metrics.height);
  ctx.stroke();
  ctx.lineWidth = 1;
}

function drawPianoRollSelection(ctx, metrics) {
  if (!state.instrumentDrag || state.instrumentDrag.mode !== "select-range") return;
  const start = Math.min(state.instrumentDrag.startBeat, state.instrumentDrag.currentBeat);
  const end = Math.max(state.instrumentDrag.startBeat, state.instrumentDrag.currentBeat);
  const x = beatToCanvasX(start, metrics);
  const width = Math.max(1, (end - start) * metrics.beatWidth);
  ctx.fillStyle = "rgba(242,184,75,0.18)";
  ctx.fillRect(x, metrics.rulerHeight, width, metrics.noteHeight);
  ctx.strokeStyle = "rgba(242,184,75,0.75)";
  ctx.strokeRect(x, metrics.rulerHeight, width, metrics.noteHeight);
}

function noteIntersectsView(note, metrics) {
  const beatStart = metrics.beatOffset;
  const beatEnd = metrics.beatOffset + metrics.visibleBeats;
  const pitchStart = metrics.pitchOffset;
  const pitchEnd = metrics.pitchOffset + metrics.visiblePitches - 1;
  return note.start + note.duration >= beatStart && note.start <= beatEnd && note.pitch >= pitchStart && note.pitch <= pitchEnd;
}

function noteAtCanvasPoint(x, y) {
  const active = activeInstrumentTrack();
  if (!active || active.kind !== "instrument") return null;
  const metrics = pianoRollMetrics();
  return [...active.notes].reverse().find((note) => {
    if (!noteIntersectsView(note, metrics)) return false;
    const nx = beatToCanvasX(note.start, metrics);
    const ny = pitchToCanvasY(note.pitch, metrics);
    const nw = Math.max(6, note.duration * metrics.beatWidth);
    const nh = Math.max(8, metrics.pitchHeight - 2);
    return x >= nx && x <= nx + nw && y >= ny && y <= ny + nh;
  }) || null;
}

function canvasPoint(event) {
  const metrics = pianoRollMetrics();
  const rect = metrics.canvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * metrics.scaleX,
    y: (event.clientY - rect.top) * metrics.scaleY,
  };
}

function quantizeBeat(value) {
  return Math.max(0, Math.round(value * 4) / 4);
}

function pitchFromY(y) {
  const metrics = pianoRollMetrics();
  return canvasYToPitch(y, metrics);
}

function setSelectedInstrumentNotes(noteIds) {
  state.selectedInstrumentNoteIds = [...new Set(noteIds)];
  state.selectedInstrumentNoteId = state.selectedInstrumentNoteIds[0] || null;
}

function activeSelectedNotes() {
  const active = activeInstrumentTrack();
  if (!active || active.kind !== "instrument") return [];
  return (active.notes || []).filter((note) => state.selectedInstrumentNoteIds.includes(note.id));
}

function selectNotesInBeatRange(startBeat, endBeat) {
  const active = activeInstrumentTrack();
  if (!active || active.kind !== "instrument") return;
  const start = Math.min(startBeat, endBeat);
  const end = Math.max(startBeat, endBeat);
  const ids = (active.notes || [])
    .filter((note) => note.start + note.duration >= start && note.start <= end)
    .map((note) => note.id);
  setSelectedInstrumentNotes(ids);
}

function copySelectedInstrumentNotes() {
  const notes = activeSelectedNotes();
  if (!notes.length) {
    showToast("Select notes to copy");
    return;
  }
  const start = Math.min(...notes.map((note) => note.start));
  state.instrumentNoteClipboard = notes.map((note) => ({ ...note, id: "", start: note.start - start }));
  showToast(`${notes.length} note${notes.length === 1 ? "" : "s"} copied`);
}

function pasteInstrumentNotes() {
  const active = activeInstrumentTrack();
  if (!active || active.kind !== "instrument") {
    showToast("Select an instrument track");
    return;
  }
  if (!state.instrumentNoteClipboard.length) {
    showToast("Copy notes first");
    return;
  }
  const pasted = state.instrumentNoteClipboard.map((note, index) => ({
    ...note,
    id: `note-${Date.now().toString(16)}-${index}`,
    start: Math.max(0, quantizeBeat((state.instrumentCursorBeat || 0) + note.start)),
  }));
  active.notes.push(...pasted);
  ensureInstrumentLengthForBeat(Math.max(...pasted.map((note) => note.start + note.duration)));
  setSelectedInstrumentNotes(pasted.map((note) => note.id));
  drawInstrumentPianoRoll();
}

function scrollPianoRoll(deltaBeats = 0, deltaPitches = 0) {
  const view = state.pianoRollView;
  view.beatOffset += deltaBeats;
  view.pitchOffset += deltaPitches;
  clampPianoRollView();
  drawInstrumentPianoRoll();
}

function zoomPianoRoll(factor, anchorBeat = null) {
  const view = state.pianoRollView;
  const oldVisible = view.visibleBeats;
  const oldOffset = view.beatOffset;
  const anchor = anchorBeat === null ? oldOffset + oldVisible / 2 : anchorBeat;
  const ratio = oldVisible > 0 ? (anchor - oldOffset) / oldVisible : 0.5;
  view.visibleBeats = oldVisible * factor;
  clampPianoRollView();
  view.beatOffset = anchor - view.visibleBeats * ratio;
  clampPianoRollView();
  drawInstrumentPianoRoll();
}

function fitPianoRoll() {
  const bounds = pianoRollContentBounds();
  state.pianoRollView.beatOffset = 0;
  state.pianoRollView.visibleBeats = Math.max(1, bounds.totalBeats);
  state.pianoRollView.pitchOffset = bounds.minPitch;
  state.pianoRollView.visiblePitches = bounds.maxPitch - bounds.minPitch + 1;
  clampPianoRollView();
  drawInstrumentPianoRoll();
}

function handlePianoRollWheel(event) {
  event.preventDefault();
  const metrics = pianoRollMetrics();
  if (event.ctrlKey || event.metaKey) {
    const point = canvasPoint(event);
    zoomPianoRoll(event.deltaY > 0 ? 1.25 : 0.8, canvasXToBeat(point.x, metrics));
    return;
  }
  if (event.shiftKey) {
    scrollPianoRoll((event.deltaY || event.deltaX) * metrics.visibleBeats / 900, 0);
    return;
  }
  scrollPianoRoll(0, event.deltaY > 0 ? -3 : 3);
}

function beginPianoRollEdit(event) {
  const point = canvasPoint(event);
  const metrics = pianoRollMetrics();
  if (point.y <= metrics.rulerHeight) {
    const beat = Math.max(0, canvasXToBeat(point.x, metrics));
    setInstrumentCursorBeat(beat);
    state.instrumentDrag = { mode: "select-range", startBeat: state.instrumentCursorBeat, currentBeat: state.instrumentCursorBeat };
    return;
  }
  const active = activeInstrumentTrack();
  if (!active || active.kind !== "instrument") {
    showToast("Select an instrument track");
    return;
  }
  const note = noteAtCanvasPoint(point.x, point.y);
  if (note) {
    const rightEdge = beatToCanvasX(note.start + note.duration, metrics);
    if (!state.selectedInstrumentNoteIds.includes(note.id)) {
      setSelectedInstrumentNotes([note.id]);
    }
    state.instrumentDrag = {
      mode: Math.abs(point.x - rightEdge) < 8 ? "resize" : "move",
      note,
      startX: point.x,
      startY: point.y,
      originalStart: note.start,
      originalDuration: note.duration,
      originalPitch: note.pitch,
      originalNotes: activeSelectedNotes().map((selected) => ({ note: selected, start: selected.start, pitch: selected.pitch })),
    };
  } else {
    const start = quantizeBeat(canvasXToBeat(point.x, metrics));
    const pitch = pitchFromY(point.y);
    const bounds = pianoRollContentBounds();
    const newNote = {
      id: `note-${Date.now().toString(16)}`,
      pitch,
      start: Math.min(start, bounds.totalBeats - 0.25),
      duration: 1,
      velocity: 0.85,
    };
    active.notes.push(newNote);
    setSelectedInstrumentNotes([newNote.id]);
    state.instrumentDrag = { mode: "resize", note: newNote, startX: point.x, startY: point.y, originalStart: newNote.start, originalDuration: 1, originalPitch: pitch };
  }
  drawInstrumentPianoRoll();
}

function movePianoRollEdit(event) {
  if (!state.instrumentDrag) return;
  const point = canvasPoint(event);
  const metrics = pianoRollMetrics();
  const drag = state.instrumentDrag;
  const bounds = pianoRollContentBounds();
  if (drag.mode === "select-range") {
    drag.currentBeat = Math.max(0, canvasXToBeat(point.x, metrics));
    setInstrumentCursorBeat(drag.currentBeat);
    selectNotesInBeatRange(drag.startBeat, drag.currentBeat);
    drawInstrumentPianoRoll();
    return;
  }
  if (drag.mode === "resize") {
    const duration = quantizeBeat(canvasXToBeat(point.x, metrics) - drag.note.start);
    drag.note.duration = Math.max(0.25, Math.min(bounds.totalBeats - drag.note.start, duration));
  } else {
    const beatDelta = quantizeBeat((point.x - drag.startX) / metrics.beatWidth);
    const pitchDelta = Math.round((drag.startY - point.y) / metrics.pitchHeight);
    const originals = drag.originalNotes && drag.originalNotes.length ? drag.originalNotes : [{ note: drag.note, start: drag.originalStart, pitch: drag.originalPitch }];
    originals.forEach((item) => {
      item.note.start = Math.max(0, Math.min(bounds.totalBeats - item.note.duration, item.start + beatDelta));
      item.note.pitch = Math.max(bounds.minPitch, Math.min(bounds.maxPitch, item.pitch + pitchDelta));
    });
  }
  drawInstrumentPianoRoll();
}

function endPianoRollEdit() {
  state.instrumentDrag = null;
}

function deleteSelectedInstrumentNote() {
  const active = activeInstrumentTrack();
  if (!active || active.kind !== "instrument") return;
  const selectedIds = state.selectedInstrumentNoteIds.length ? state.selectedInstrumentNoteIds : [state.selectedInstrumentNoteId].filter(Boolean);
  if (!selectedIds.length) return;
  active.notes = active.notes.filter((note) => !selectedIds.includes(note.id));
  setSelectedInstrumentNotes([]);
  drawInstrumentPianoRoll();
}

const KEYBOARD_NOTE_OFFSETS = {
  a: 0,
  w: 1,
  s: 2,
  e: 3,
  d: 4,
  f: 5,
  t: 6,
  g: 7,
  y: 8,
  h: 9,
  u: 10,
  j: 11,
  k: 12,
};

async function playKeyboardPitch(pitch) {
  const context = ensureInstrumentAudioContext();
  context.resume();
  const track = activeInstrumentTrack() || { instrument: el.instrumentPatch.value, volume: 0.85 };
  await prepareInstrumentSamples(context, [track]);
  scheduleSynthNote(context, context.destination, { pitch, start: 0, duration: 0.5, velocity: 0.9 }, track, context.currentTime);
  if (state.instrumentRecording && track.kind === "instrument") {
    const transportBeat = transportBeatAtCurrentTime();
    if (transportBeat === null || transportBeat < 0) return;
    const start = Math.max(0, quantizeBeat(transportBeat));
    ensureInstrumentLengthForBeat(start + 1);
    track.notes.push({ id: `note-${Date.now().toString(16)}`, pitch, start, duration: 1, velocity: 0.9 });
    drawInstrumentPianoRoll();
  }
}

function handleInstrumentKeydown(event) {
  if (!el.instrumentLabPage.classList.contains("active")) return;
  if (event.target && ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)) return;
  const key = event.key.toLowerCase();
  if (!(key in KEYBOARD_NOTE_OFFSETS)) return;
  event.preventDefault();
  const pitch = Number(el.instrumentOctave.value || 4) * 12 + 12 + KEYBOARD_NOTE_OFFSETS[key];
  playKeyboardPitch(pitch).catch((error) => showToast(error.message));
}

function renderInstrumentPianoKeys() {
  el.instrumentPianoKeys.replaceChildren();
  for (let offset = 0; offset <= 12; offset += 1) {
    const pitch = Number(el.instrumentOctave.value || 4) * 12 + 12 + offset;
    const button = document.createElement("button");
    button.type = "button";
    button.className = `piano-key ${[1, 3, 6, 8, 10].includes(offset % 12) ? "black" : "white"}`;
    button.textContent = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "C"][offset];
    button.addEventListener("pointerdown", () => {
      playKeyboardPitch(pitch).catch((error) => showToast(error.message));
    });
    el.instrumentPianoKeys.appendChild(button);
  }
}

function addGeneratedResult(result, plan) {
  state.generatedResults.unshift({ result, plan });
  state.generatedResults = state.generatedResults.slice(0, 12);
  renderGeneratedList();
}

async function loadAll() {
  await loadInstrumentBank();
  const [status, runtime, presets, models, tracks, extractions, musicGenerations, instrumentClips, editorAssets, logs] = await Promise.all([
    api("/api/status"),
    api("/api/runtime/status"),
    api("/api/presets"),
    api("/api/models"),
    api("/api/extractions/tracks"),
    api("/api/extractions"),
    api("/api/music-generations"),
    api("/api/instrument-lab/clips"),
    api("/api/editor/assets"),
    api("/api/logs"),
  ]);
  state.presets = presets;
  state.models = models;
  state.extractionTracks = tracks;
  state.extractionResults = extractions;
  state.musicResults = musicGenerations;
  state.instrumentClips = instrumentClips;
  state.editorAssets = editorAssets;
  renderStatus(status);
  renderRuntime(runtime);
  renderPresets();
  renderModels();
  renderExtractionTracks();
  renderExtractionList();
  applyMusicModelDefaults();
  syncMusicVocalControls();
  renderMusicList();
  renderInstrumentTracks();
  renderInstrumentPianoKeys();
  drawInstrumentPianoRoll();
  renderInstrumentClipList();
  renderSourceAssetOptions();
  renderEditorAssets();
  renderLogs(logs);
}

function numericValue(node) {
  return node.value === "" ? null : Number(node.value);
}

function currentSettings() {
  return {
    contextSeconds: numericValue(el.contextSeconds),
    newSeconds: numericValue(el.newSeconds),
    repaintOverlapSeconds: numericValue(el.repaintOverlapSeconds),
  };
}

function updateSelectionReadout() {
  const continuation = Number(el.continuationSlider.value || 0);
  const settings = currentSettings();
  const context = settings.contextSeconds || 0;
  const future = settings.newSeconds || 0;
  const repaintBefore = settings.repaintOverlapSeconds || 0;
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
  el.contextRange.textContent = `Source context: ${formatTime(start)} to ${formatTime(continuation)} (${tail.toFixed(1)}s), repaint starts ${repaintBefore.toFixed(1)}s before marker`;
  setPill(el.sourceState, "Source loaded", "ok");
}

function aceStepSettingsPayload() {
  return {
    inference_steps: numericValue(el.inferenceSteps),
    guidance_scale: numericValue(el.guidanceScale),
    shift: numericValue(el.shiftValue),
    repaint_strength: numericValue(el.repaintStrength),
    repaint_mode: el.repaintMode.value || null,
    repaint_latent_crossfade_frames: numericValue(el.repaintLatentCrossfadeFrames),
    repaint_wav_crossfade_sec: numericValue(el.repaintWavCrossfadeSec),
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

async function useGeneratedAsSource(sourcePath) {
  setPill(el.sourceState, "Loading", "warn");
  try {
    const probe = await api("/api/source/probe", {
      method: "POST",
      body: JSON.stringify({ source_path: sourcePath }),
    });
    await loadProbeIntoPlayer(sourcePath, probe);
    el.selectedFileName.textContent = "Generated output";
    showToast("Generated output loaded as source");
  } catch (error) {
    setPill(el.sourceState, "Error", "error");
    showToast(error.message);
  } finally {
    refreshLogs();
  }
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

async function loadExtractionProbeIntoPlayer(sourcePath, probe) {
  state.extractSourceProbe = probe;
  el.extractSourcePath.value = sourcePath;
  el.extractSourceAudio.src = `/api/extractions/audio?path=${encodeURIComponent(sourcePath)}`;
  el.extractSourceDuration.textContent = `Duration ${formatTime(probe.duration_seconds)}`;
  el.extractSourceFormatReadout.textContent = `Source format: ${probe.source_format}; full song extraction`;
  setPill(el.extractSourceState, "Source loaded", "ok");
}

async function loadExtractionSource() {
  setPill(el.extractSourceState, "Loading", "warn");
  el.loadExtractSourceButton.disabled = true;
  try {
    const sourcePath = el.extractSourcePath.value.trim();
    const probe = await api("/api/extractions/source/probe", {
      method: "POST",
      body: JSON.stringify({ source_path: sourcePath }),
    });
    await loadExtractionProbeIntoPlayer(sourcePath, probe);
    showToast("Extraction source loaded");
  } catch (error) {
    state.extractSourceProbe = null;
    setPill(el.extractSourceState, "Error", "error");
    showToast(error.message);
  } finally {
    el.loadExtractSourceButton.disabled = false;
    refreshLogs();
  }
}

async function uploadExtractionSourceFile() {
  const file = el.extractSourceFile.files && el.extractSourceFile.files[0];
  if (!file) return;

  setPill(el.extractSourceState, "Uploading", "warn");
  el.extractSelectedFileName.textContent = file.name;
  el.loadExtractSourceButton.disabled = true;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/api/extractions/source/upload", {
      method: "POST",
      body: formData,
    });
    const body = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(body && body.detail ? body.detail : `Upload failed: ${response.status}`);
    }

    await loadExtractionProbeIntoPlayer(body.stored_path, body.probe);
    showToast("Extraction source loaded");
  } catch (error) {
    state.extractSourceProbe = null;
    setPill(el.extractSourceState, "Error", "error");
    showToast(error.message);
  } finally {
    el.loadExtractSourceButton.disabled = false;
    refreshLogs();
  }
}

async function runExtraction() {
  setPill(el.extractActionState, "Extracting", "warn");
  el.extractionActivity.innerHTML = "<strong>Starting</strong><br>Preparing ACE-Step extract request.";
  el.runExtractionButton.disabled = true;
  try {
    const response = await api("/api/extractions/run", {
      method: "POST",
      body: JSON.stringify({
        source_path: el.extractSourcePath.value.trim(),
        track_name: el.extractTrackSelect.value,
        label: el.extractLabelInput.value.trim() || null,
        output_format: el.extractOutputFormat.value,
        inference_steps: numericValue(el.extractInferenceSteps),
        guidance_scale: numericValue(el.extractGuidanceScale),
        shift: numericValue(el.extractShift),
        seed: numericValue(el.extractSeedInput),
        instruction: el.extractInstruction.value.trim() || null,
      }),
    });
    state.extractionResults.unshift(response.extraction);
    state.extractionResults = state.extractionResults.slice(0, 24);
    renderExtractionList();
    await refreshEditorAssets();
    if (response.extraction.status === "complete") {
      setPill(el.extractActionState, "Complete", "ok");
      el.extractionActivity.innerHTML = "<strong>Complete</strong><br>Track extraction finished.";
    } else {
      setPill(el.extractActionState, "Failed", "error");
      el.extractionActivity.innerHTML = `<strong>Failed</strong><br>${response.extraction.message}`;
    }
    showToast(response.extraction.message);
  } catch (error) {
    setPill(el.extractActionState, "Error", "error");
    el.extractionActivity.innerHTML = `<strong>Error</strong><br>${error.message}`;
    showToast(error.message);
  } finally {
    el.runExtractionButton.disabled = false;
    refreshLogs();
  }
}

async function runMusicGeneration() {
  const prompt = el.musicPrompt.value.trim();
  if (!prompt) {
    showToast("Enter a music prompt");
    return;
  }
  setPill(el.musicActionState, "Generating", "warn");
  el.musicActivity.innerHTML = "<strong>Starting</strong><br>Preparing ACE-Step text-to-music request.";
  el.runMusicButton.disabled = true;
  try {
    const response = await api("/api/music-generations/run", {
      method: "POST",
      body: JSON.stringify({
        prompt,
        model: el.musicModelSelect.value,
        label: el.musicLabelInput.value.trim() || null,
        instrumental: el.musicInstrumental.checked,
        lyrics: el.musicLyrics.value.trim() || null,
        vocal_language: el.musicVocalLanguage.value,
        output_format: el.musicOutputFormat.value,
        audio_duration: numericValue(el.musicDuration),
        inference_steps: numericValue(el.musicInferenceSteps),
        guidance_scale: numericValue(el.musicGuidanceScale),
        shift: numericValue(el.musicShift),
        infer_method: el.musicInferMethod.value,
        use_tiled_decode: el.musicUseTiledDecode.checked,
        dcw_enabled: el.musicDcwEnabled.checked,
        velocity_norm_threshold: numericValue(el.musicVelocityNormThreshold),
        velocity_ema_factor: numericValue(el.musicVelocityEmaFactor),
        seed: numericValue(el.musicSeed),
      }),
    });
    state.musicResults.unshift(response.generation);
    state.musicResults = state.musicResults.slice(0, 24);
    renderMusicList();
    await refreshEditorAssets();
    if (response.generation.status === "complete") {
      setPill(el.musicActionState, "Complete", "ok");
      el.musicActivity.innerHTML = "<strong>Complete</strong><br>Music generation finished.";
    } else {
      setPill(el.musicActionState, "Failed", "error");
      el.musicActivity.innerHTML = `<strong>Failed</strong><br>${escapeHtml(response.generation.message)}`;
    }
    showToast(response.generation.message);
  } catch (error) {
    setPill(el.musicActionState, "Error", "error");
    el.musicActivity.innerHTML = `<strong>Error</strong><br>${escapeHtml(error.message)}`;
    showToast(error.message);
  } finally {
    el.runMusicButton.disabled = false;
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
      preset: state.selectedPreset ? state.selectedPreset.slug : "smooth-continuation",
      model_slug: state.selectedModel ? state.selectedModel.slug : "acestep-v15-turbo",
      auto_install: el.autoInstallModel.checked,
      caption: el.captionInput.value.trim(),
      output_dir: el.outputDir.value.trim() || null,
      context_seconds: numericValue(el.contextSeconds),
      repaint_overlap_seconds: numericValue(el.repaintOverlapSeconds),
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
    await refreshEditorAssets();
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
  const selectedSlug = state.selectedModel && state.selectedModel.slug;
  renderModels();
  const selected = state.models.find((model) => model.slug === selectedSlug);
  if (selected) {
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
  await refreshEditorAssets();
  await refreshLogs();
  showToast("Status refreshed");
}

el.transitionTabButton.addEventListener("click", () => setActivePage("transition"));
el.extractionTabButton.addEventListener("click", () => setActivePage("extraction"));
el.musicTabButton.addEventListener("click", () => setActivePage("music"));
el.instrumentLabTabButton.addEventListener("click", () => setActivePage("instrument"));
el.audioEditTabButton.addEventListener("click", () => setActivePage("audioedit"));
el.reloadAudioEditorButton.addEventListener("click", reloadAudioEditor);
el.openAudioEditorButton.addEventListener("click", openAudioEditorWindow);
el.refreshEditorAssetsButton.addEventListener("click", async () => {
  await refreshEditorAssets();
  showToast("Editor assets refreshed");
});
el.editorAssetSearch.addEventListener("input", renderEditorAssets);
el.editorCategoryFilter.addEventListener("change", renderEditorAssets);
el.editSaveFile.addEventListener("change", () => {
  const file = el.editSaveFile.files && el.editSaveFile.files[0];
  el.editSaveFileName.textContent = file ? file.name : "No file selected";
});
el.saveEditButton.addEventListener("click", saveEditedAudio);
el.generateButton.addEventListener("click", generateTransition);
el.loadSourceAssetButton.addEventListener("click", loadExistingCreationAsTransitionSource);
el.loadSourceButton.addEventListener("click", loadSource);
el.sourceFile.addEventListener("change", uploadSourceFile);
el.loadExtractSourceAssetButton.addEventListener("click", loadExistingCreationAsExtractionSource);
el.loadExtractSourceButton.addEventListener("click", loadExtractionSource);
el.extractSourceFile.addEventListener("change", uploadExtractionSourceFile);
el.runExtractionButton.addEventListener("click", runExtraction);
el.mergeExtractionsButton.addEventListener("click", mergeSelectedExtractions);
el.runMusicButton.addEventListener("click", runMusicGeneration);
el.musicModelSelect.addEventListener("change", applyMusicModelDefaults);
el.musicInstrumental.addEventListener("change", syncMusicVocalControls);
el.addInstrumentTrackButton.addEventListener("click", addInstrumentTrack);
el.importInstrumentAssetButton.addEventListener("click", importInstrumentAssetTrack);
el.playInstrumentButton.addEventListener("click", playInstrumentLab);
el.stopInstrumentButton.addEventListener("click", stopInstrumentLab);
el.recordInstrumentButton.addEventListener("click", () => {
  if (state.instrumentRecording) {
    stopInstrumentLab();
    return;
  }
  recordInstrumentLab().catch(() => {});
});
el.deleteInstrumentNoteButton.addEventListener("click", deleteSelectedInstrumentNote);
el.copyInstrumentNotesButton.addEventListener("click", copySelectedInstrumentNotes);
el.pasteInstrumentNotesButton.addEventListener("click", pasteInstrumentNotes);
el.renderInstrumentButton.addEventListener("click", () => {
  renderInstrumentPreview().catch(() => {});
});
el.saveInstrumentTrackButton.addEventListener("click", () => saveInstrumentClip({ trackOnly: true }));
el.saveInstrumentButton.addEventListener("click", saveInstrumentClip);
el.instrumentPatch.addEventListener("change", () => {
  const active = activeInstrumentTrack();
  if (active && active.kind === "instrument") {
    active.instrument = legacyInstrumentId(el.instrumentPatch.value);
    renderInstrumentTracks();
  }
  updateInstrumentInfo();
});
el.importSfzButton.addEventListener("click", importSfzInstrument);
el.instrumentBars.addEventListener("change", () => {
  clampPianoRollView();
  drawInstrumentPianoRoll();
});
el.instrumentBpm.addEventListener("change", drawInstrumentPianoRoll);
el.instrumentOctave.addEventListener("change", renderInstrumentPianoKeys);
el.pianoRollScroll.addEventListener("input", (event) => {
  state.pianoRollView.beatOffset = Number(event.target.value || 0);
  clampPianoRollView();
  drawInstrumentPianoRoll();
});
el.pianoRollZoomOutButton.addEventListener("click", () => zoomPianoRoll(1.25));
el.pianoRollZoomInButton.addEventListener("click", () => zoomPianoRoll(0.8));
el.pianoRollFitButton.addEventListener("click", fitPianoRoll);
el.instrumentPianoRoll.addEventListener("pointerdown", beginPianoRollEdit);
el.instrumentPianoRoll.addEventListener("pointermove", movePianoRollEdit);
el.instrumentPianoRoll.addEventListener("wheel", handlePianoRollWheel, { passive: false });
window.addEventListener("pointerup", endPianoRollEdit);
window.addEventListener("keydown", handleInstrumentKeydown);
el.refreshMusicButton.addEventListener("click", async () => {
  await refreshMusicGenerations();
  await refreshEditorAssets();
  showToast("Music generations refreshed");
});
el.refreshExtractionsButton.addEventListener("click", async () => {
  await refreshExtractions();
  await refreshEditorAssets();
  showToast("Extractions refreshed");
});
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

[el.contextSeconds, el.newSeconds, el.repaintOverlapSeconds].forEach((node) => {
  node.addEventListener("input", updateSelectionReadout);
});

[
  el.inferenceSteps,
  el.guidanceScale,
  el.shiftValue,
  el.repaintStrength,
  el.repaintMode,
  el.repaintLatentCrossfadeFrames,
  el.repaintWavCrossfadeSec,
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
