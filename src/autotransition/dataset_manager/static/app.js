const state = {
  songs: [], selected: null, kind: 'vocals', selectedWord: null,
  selectedPhone: null, audio: null, waveform: null, waveformKind: null,
  poller: null, zoom: 1, drag: null, justDragged: false,
  timelineDurationMs: 0,
};

const $ = (selector, root = document) => root.querySelector(selector);
const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const ms = value => Number.isFinite(Number(value)) ? Number(value) : 0;
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const formatDuration = value => { const seconds = Math.round(ms(value) / 1000); const m = Math.floor(seconds / 60); return `${m}:${String(seconds % 60).padStart(2,'0')}`; };
const statusClass = value => String(value || '').toLowerCase().replaceAll(' ','-');

async function api(path, options = {}) {
  const response = await fetch(path, { headers: {'Content-Type':'application/json', ...(options.headers || {})}, ...options });
  if (!response.ok) { let message = `${response.status} ${response.statusText}`; try { const body = await response.json(); message = body.detail || message; } catch {} throw new Error(message); }
  return response.json();
}

async function refresh() {
  const [stats, songs] = await Promise.all([
    api('/api/dataset/stats'),
    api(`/api/dataset/songs?search=${encodeURIComponent($('#search').value)}&status=${encodeURIComponent($('#statusFilter').value)}`)
  ]);
  renderStats(stats);
  state.songs = songs;
  if (state.selected) {
    const updated = songs.find(item => item.id === state.selected.id);
    state.selected = updated ? await api(`/api/dataset/songs/${updated.id}`) : null;
  }
  renderSongs();
}

function renderStats(stats) {
  const values = [['songs','Songs'],['duration_ms','Imported'],['saved_songs','Saved'],['approved_duration_ms','Approved time'],['awaiting_review','Needs review'],['active_jobs','Active jobs']];
  $('#stats').innerHTML = values.map(([key,label]) => `<div class="stat"><strong>${key.includes('duration') ? formatDuration(stats[key]) : stats[key]}</strong><span>${label}</span></div>`).join('');
}

function renderSongs() {
  const list = $('#songList');
  if (!state.songs.length) { list.innerHTML = '<div class="empty">No songs imported yet.</div>'; return; }
  list.innerHTML = state.songs.map(song => {
    const active = state.selected && state.selected.id === song.id;
    const detail = active ? editorHtml(state.selected) : '';
    return `<article class="song-row"><div class="song-summary">
      <div><div class="song-title">${escapeHtml(song.title)}</div><div class="song-meta">${escapeHtml(song.artist || song.source_filename)} · ${escapeHtml(song.source_filename)}</div></div>
      <div class="song-meta">${formatDuration(song.duration_ms)}<br>${escapeHtml(song.language)}</div>
      <div><span class="state ${statusClass(song.status)}">${escapeHtml(song.status)}</span>${song.has_unsaved_changes ? '<div class="song-meta">Unsaved changes</div>' : ''}</div>
      <div class="song-meta">${song.last_error ? `<span class="error">${escapeHtml(song.last_error.slice(0,140))}</span>` : 'No extraction yet'}</div>
      <div class="row-actions"><button class="button secondary small open-song" data-id="${song.id}">${active ? 'Close' : 'Open'}</button><button class="button primary small extract-song" data-id="${song.id}">${song.status === 'Failed' ? 'Retry' : 'Extract'}</button></div>
    </div>${detail}</article>`;
  }).join('');
  list.querySelectorAll('.open-song').forEach(button => button.addEventListener('click', () => openSong(button.dataset.id)));
  list.querySelectorAll('.extract-song').forEach(button => button.addEventListener('click', () => extractSong(button.dataset.id)));
  if (state.selected) wireEditor();
}

function editorHtml(song) {
  const draft = song.draft?.annotation;
  const run = song.run;
  const words = draft?.words || [];
  const phoneCount = words.reduce((sum, word) => sum + (word.phonemes || []).length, 0);
  const warning = draft?.quality?.warnings?.length ? `<div class="warning">${draft.quality.warnings.map(escapeHtml).join(' · ')}</div>` : '';
  const runMessage = run && run.status !== 'succeeded' ? `<div class="${run.status === 'failed' ? 'error' : 'muted'}">${escapeHtml(run.current_stage || run.status)} · ${Math.round((run.progress || 0) * 100)}% ${escapeHtml(run.error || '')}</div>` : '';
  return `<div class="editor" data-song="${song.id}">
    <div class="editor-head"><div><h3>${escapeHtml(song.title)}</h3><p class="muted">${escapeHtml(song.source_filename)} · ${formatDuration(song.duration_ms)} · ${words.length} words · ${phoneCount} phonemes</p>${runMessage}${warning}</div><div class="editor-actions"><button class="button secondary small" id="editLyrics">Re-enter lyrics</button><button class="button secondary small" id="saveDraft">Save draft</button><button class="button primary small" id="approveRevision">Save approved revision</button></div></div>
    <div class="player-row"><audio id="audio" controls preload="metadata"></audio><div class="source-toggle"><button data-kind="vocals" class="${state.kind === 'vocals' ? 'active' : ''}">Vocals</button><button data-kind="original" class="${state.kind === 'original' ? 'active' : ''}">Original</button></div></div>
    <div class="timeline-tools"><div class="timeline-tool-group"><span class="muted">Timeline</span><button class="icon-button timeline-button" id="zoomOut" title="Zoom out" aria-label="Zoom out">−</button><span id="zoomLabel" class="zoom-label">100%</span><button class="icon-button timeline-button" id="zoomIn" title="Zoom in" aria-label="Zoom in">+</button><button class="button secondary small" id="zoomFit">Fit</button></div><span class="muted">Audio-relative timing</span></div>
    <div class="timeline-viewport" id="timelineViewport"><div class="timeline" id="timeline"><div class="timeline-content" id="timelineContent">
      <canvas id="wave" class="wave"></canvas>
      <div class="lane" id="wordsLane"><div class="lane-head"><span class="lane-label">Words</span><button class="lane-action" id="removeWordRow" title="Remove the selected word">Remove selected</button></div><div class="lane-track">${words.map((word,index) => annotationBox(word,index,false,song.duration_ms)).join('')}</div></div>
      <div class="lane phoneme-lane" id="phonesLane"><div class="lane-head"><span class="lane-label">Phones</span><button class="lane-action" id="removePhoneRow" title="Remove the selected phoneme">Remove selected</button></div><div class="lane-track">${words.flatMap((word,wi) => (word.phonemes || []).map((phone,pi) => annotationBox(phone,`${wi}:${pi}`,true,song.duration_ms))).join('')}</div></div>
      <div class="playhead" id="playhead"></div>
    </div></div></div>
    <div class="inspector" id="inspector"><div class="wide muted">Select a word or phoneme to edit its exact audio-relative boundaries.</div></div>
    <div class="editor-foot"><button class="button secondary small" id="addWord">Add word at playhead</button><button class="button secondary small" id="addPhone">Add phone at playhead</button><button class="button secondary small" id="deleteSelected">Delete selected</button><button class="button secondary small" id="exportSong">Export approved</button><span class="muted">${escapeHtml(draft?.phone_timing_status || 'alignment pending')}</span></div>
  </div>`;
}

function annotationBox(item, index, phone, duration) {
  const start = Math.max(0, ms(item.start_ms));
  const end = Math.max(start + 1, ms(item.end_ms));
  const left = duration ? (start / duration) * 100 : 0;
  const width = duration ? ((end - start) / duration) * 100 : 0;
  const selected = phone ? state.selectedPhone === index : state.selectedPhone === null && state.selectedWord === index;
  return `<button class="annotation ${phone ? 'phoneme' : ''} ${selected ? 'selected' : ''}" style="left:${left}%;width:${width}%" data-kind="${phone ? 'phone' : 'word'}" data-index="${index}" title="${start.toFixed(1)}–${end.toFixed(1)} ms"><span class="annotation-handle annotation-handle-start" data-edge="start" aria-hidden="true"></span><span class="annotation-label">${escapeHtml(phone ? item.symbol : item.text)}</span><span class="annotation-handle annotation-handle-end" data-edge="end" aria-hidden="true"></span></button>`;
}

async function openSong(id) {
  state.selected = await api(`/api/dataset/songs/${id}`);
  state.selectedWord = null; state.selectedPhone = null; state.zoom = 1;
  state.waveform = null; state.waveformKind = null; state.timelineDurationMs = state.selected.duration_ms;
  renderSongs();
}

async function extractSong(id) {
  try { await api(`/api/dataset/songs/${id}/extract`, {method:'POST'}); await openSong(id); startPolling(); }
  catch (error) { alert(error.message); }
}

function startPolling() {
  clearInterval(state.poller);
  state.poller = setInterval(async () => {
    try { await refresh(); if (!state.songs.some(song => ['Queued','Extracting'].includes(song.status))) clearInterval(state.poller); }
    catch {}
  }, 1800);
}

function timelineDuration() { return state.timelineDurationMs || state.selected?.duration_ms || 1; }

function timelineMetrics() {
  const viewport = $('#timelineViewport'); const duration = timelineDuration();
  const viewportWidth = Math.max(1, viewport?.clientWidth || 900);
  const fitPixelsPerSecond = viewportWidth / Math.max(.001, duration / 1000);
  const pixelsPerSecond = fitPixelsPerSecond * state.zoom;
  return { duration, viewportWidth, pixelsPerSecond, width: Math.max(viewportWidth, duration / 1000 * pixelsPerSecond) };
}

function updateTimelineGeometry() {
  const content = $('#timelineContent'); if (!content) return;
  const metrics = timelineMetrics(); content.style.width = `${metrics.width}px`; content.dataset.pixelsPerSecond = String(metrics.pixelsPerSecond);
  const label = $('#zoomLabel'); if (label) label.textContent = `${Math.round(state.zoom * 100)}%`;
  drawWaveform(); updatePlayhead();
}

function updateAnnotationVisuals() {
  const annotation = state.selected?.draft?.annotation;
  const duration = timelineDuration();
  if (!annotation || !duration) return;
  document.querySelectorAll('.annotation').forEach(button => {
    const isPhone = button.dataset.kind === 'phone';
    let item = null;
    if (isPhone) {
      const [wordIndex, phoneIndex] = button.dataset.index.split(':').map(Number);
      item = annotation.words?.[wordIndex]?.phonemes?.[phoneIndex] || null;
    } else {
      item = annotation.words?.[Number(button.dataset.index)] || null;
    }
    if (!item) return;
    const start = Math.max(0, ms(item.start_ms));
    const end = Math.max(start + 1, ms(item.end_ms));
    button.style.left = `${(start / duration) * 100}%`;
    button.style.width = `${((end - start) / duration) * 100}%`;
    button.title = `${start.toFixed(1)}–${end.toFixed(1)} ms`;
  });
}

async function loadAudio() {
  if (!state.selected) return;
  const audio = $('#audio'); if (!audio) return;
  state.audio = audio;
  audio.onloadedmetadata = () => { state.timelineDurationMs = Number.isFinite(audio.duration) ? audio.duration * 1000 : state.selected.duration_ms; updateTimelineGeometry(); updateAnnotationVisuals(); };
  audio.ontimeupdate = updatePlayhead;
  const source = `/api/dataset/songs/${state.selected.id}/audio?kind=${state.kind}`;
  if (audio.dataset.source !== source) { audio.dataset.source = source; audio.src = source; audio.load(); }
  updateTimelineGeometry();
}

async function loadWaveform() {
  if (!state.selected || (state.waveform && state.waveformKind === state.kind)) { drawWaveform(); return; }
  try { state.waveform = await api(`/api/dataset/songs/${state.selected.id}/waveform?kind=${state.kind}`); state.waveformKind = state.kind; drawWaveform(); }
  catch { state.waveform = null; drawWaveform(); }
}

function drawWaveform() {
  const canvas = $('#wave'); const content = $('#timelineContent'); if (!canvas || !content) return;
  const width = Math.max(1, content.clientWidth); const height = 100; const scale = window.devicePixelRatio || 1;
  canvas.width = width * scale; canvas.height = height * scale; canvas.style.width = `${width}px`; canvas.style.height = `${height}px`;
  const context = canvas.getContext('2d'); context.setTransform(scale,0,0,scale,0,0); context.clearRect(0,0,width,height); context.strokeStyle = '#67d3b4'; context.globalAlpha = .75; context.lineWidth = 1;
  const data = state.waveform?.peaks || []; if (!data.length) return;
  const mid = height / 2; context.beginPath(); data.forEach((peak,index) => { const x = index / Math.max(1,data.length-1) * width; const bar = Math.max(1, peak * height * .8); context.moveTo(x,mid-bar/2); context.lineTo(x,mid+bar/2); }); context.stroke();
}

function updatePlayhead() {
  const head = $('#playhead'); if (!head || !state.audio) return;
  const ratio = state.audio.currentTime * 1000 / timelineDuration(); head.style.left = `${clamp(ratio,0,1) * 100}%`; head.style.display = 'block';
}

function itemForSelection() {
  const annotation = state.selected?.draft?.annotation; if (!annotation) return null;
  if (state.selectedPhone !== null) { const [wi,pi] = state.selectedPhone.split(':').map(Number); return annotation.words?.[wi]?.phonemes?.[pi] || null; }
  if (state.selectedWord !== null) return annotation.words?.[state.selectedWord] || null;
  return null;
}

function selectionKey(kind, index) { return kind === 'phone' ? `phone:${index}` : `word:${index}`; }

function selectAnnotation(kind, index, seek = true) {
  state.selectedPhone = kind === 'phone' ? index : null; state.selectedWord = kind === 'phone' ? Number(index.split(':')[0]) : Number(index);
  document.querySelectorAll('.annotation').forEach(button => button.classList.toggle('selected', selectionKey(button.dataset.kind, button.dataset.index) === selectionKey(kind, index)));
  renderInspector();
  if (seek && state.audio) { const item = itemForSelection(); if (item) state.audio.currentTime = ms(item.start_ms) / 1000; }
}

function wordForSelection() {
  if (state.selectedWord === null || !state.selected?.draft?.annotation?.words) return null;
  return state.selected.draft.annotation.words[state.selectedWord] || null;
}

function markDraftDirty() {
  if (!state.selected?.draft?.annotation) return;
  state.selected.draft.annotation.quality = {...(state.selected.draft.annotation.quality || {}), approved:false};
}

function updateWordChildren(word, originalStart, originalEnd, newStart, newEnd, originalPhones, mode) {
  const oldDuration = Math.max(1, originalEnd - originalStart); const newDuration = Math.max(1, newEnd - newStart);
  (word.phonemes || []).forEach((phone, index) => { const original = originalPhones[index]; if (!original) return; if (mode === 'move') { const delta = newStart - originalStart; phone.start_ms = original.start_ms + delta; phone.end_ms = original.end_ms + delta; } else { phone.start_ms = newStart + (original.start_ms - originalStart) * newDuration / oldDuration; phone.end_ms = newStart + (original.end_ms - originalStart) * newDuration / oldDuration; } });
}

function beginDrag(event, button, edgeOverride = null) {
  if (!state.selected?.draft) return;
  event.preventDefault(); const kind = button.dataset.kind; const index = button.dataset.index; selectAnnotation(kind, index, false);
  if (button.setPointerCapture) button.setPointerCapture(event.pointerId);
  const item = itemForSelection(); if (!item) return; const rect = button.getBoundingClientRect(); const offset = event.clientX - rect.left; const edge = Math.min(10, rect.width / 3);
  let mode = edgeOverride === 'start' ? 'resize-start' : edgeOverride === 'end' ? 'resize-end' : 'move';
  if (!edgeOverride) { if (offset <= edge) mode = 'resize-start'; else if (rect.width - offset <= edge) mode = 'resize-end'; }
  const word = kind === 'phone' ? wordForSelection() : item;
  state.drag = {button,pointerId:event.pointerId,kind,index,mode,startX:event.clientX,moved:false,originalStart:ms(item.start_ms),originalEnd:ms(item.end_ms),originalPhones:word ? (word.phonemes || []).map(phone => ({start_ms:ms(phone.start_ms),end_ms:ms(phone.end_ms)})) : []};
}

function handleDragMove(event) {
  const drag = state.drag; if (!drag || !state.selected?.draft) return;
  if (drag.pointerId !== event.pointerId) return;
  const content = $('#timelineContent'); const pixelsPerSecond = Number(content?.dataset.pixelsPerSecond || 0); if (!pixelsPerSecond) return;
  const delta = (event.clientX - drag.startX) / pixelsPerSecond * 1000; if (Math.abs(event.clientX - drag.startX) > 2) drag.moved = true;
  const duration = timelineDuration(); const annotation = state.selected.draft.annotation; const item = drag.kind === 'phone' ? itemForSelection() : annotation.words[Number(drag.index)]; if (!item) return;
  if (drag.kind === 'word') {
    const originalDuration = drag.originalEnd - drag.originalStart; let start = drag.originalStart; let end = drag.originalEnd;
    if (drag.mode === 'move') { const previousEnd = Number(drag.index) > 0 ? ms(annotation.words[Number(drag.index)-1].end_ms) : 0; const nextStart = Number(drag.index) + 1 < annotation.words.length ? ms(annotation.words[Number(drag.index)+1].start_ms) : duration; start = clamp(drag.originalStart + delta, previousEnd, Math.max(previousEnd, nextStart - originalDuration)); end = start + originalDuration; }
    else if (drag.mode === 'resize-start') start = clamp(drag.originalStart + delta, 0, drag.originalEnd - 1); else end = clamp(drag.originalEnd + delta, drag.originalStart + 1, duration);
    item.start_ms = start; item.end_ms = end; updateWordChildren(item, drag.originalStart, drag.originalEnd, start, end, drag.originalPhones, drag.mode === 'move' ? 'move' : 'resize');
  } else {
    const parent = wordForSelection(); if (!parent) return; const parentStart = ms(parent.start_ms); const parentEnd = ms(parent.end_ms); const originalDuration = drag.originalEnd - drag.originalStart; let start = drag.originalStart; let end = drag.originalEnd;
    if (drag.mode === 'move') { start = clamp(drag.originalStart + delta, parentStart, parentEnd - originalDuration); end = start + originalDuration; } else if (drag.mode === 'resize-start') start = clamp(drag.originalStart + delta, parentStart, drag.originalEnd - 1); else end = clamp(drag.originalEnd + delta, drag.originalStart + 1, parentEnd);
    item.start_ms = start; item.end_ms = end;
  }
  markDraftDirty(); updateAnnotationVisuals(); renderInspector();
}

function endDrag(event) { if (!state.drag || (event?.pointerId !== undefined && state.drag.pointerId !== event.pointerId)) return; const drag = state.drag; const moved = drag.moved; const item = itemForSelection(); if (drag.button?.hasPointerCapture?.(drag.pointerId)) drag.button.releasePointerCapture(drag.pointerId); state.justDragged = moved; state.drag = null; renderInspector(); if (!moved && item && state.audio) state.audio.currentTime = ms(item.start_ms) / 1000; }

function renderInspector() {
  const inspector = $('#inspector'); if (!inspector) return; const item = itemForSelection();
  if (!item) { inspector.innerHTML = '<div class="wide muted">Select a word or phoneme to edit its exact audio-relative boundaries.</div>'; return; }
  const phone = state.selectedPhone !== null;
  inspector.innerHTML = `${phone ? `<label class="wide">Phoneme<input id="editText" value="${escapeHtml(item.symbol)}"></label>` : `<label class="wide">Word<input id="editText" value="${escapeHtml(item.text)}"></label>`}<label>Start (ms)<input id="editStart" type="number" step="1" value="${ms(item.start_ms).toFixed(1)}"></label><label>End (ms)<input id="editEnd" type="number" step="1" value="${ms(item.end_ms).toFixed(1)}"></label><div class="inspector-actions"><button class="button danger small" id="deleteItem">Delete ${phone ? 'phone' : 'word'}</button><button class="button primary small" id="applyItem">Apply</button></div>`;
  $('#applyItem').addEventListener('click', applyItem); $('#deleteItem').addEventListener('click', deleteItem);
}

function applyItem() {
  const item = itemForSelection(); if (!item) return; const requestedStart = ms($('#editStart').value); const requestedEnd = ms($('#editEnd').value);
  if (state.selectedPhone !== null) { const parent = wordForSelection(); if (!parent) return; item.symbol = $('#editText').value.trim() || item.symbol; item.start_ms = clamp(requestedStart, ms(parent.start_ms), ms(parent.end_ms) - 1); item.end_ms = clamp(requestedEnd, item.start_ms + 1, ms(parent.end_ms)); }
  else { const oldStart = ms(item.start_ms); const oldEnd = ms(item.end_ms); const start = clamp(requestedStart, 0, timelineDuration() - 1); const end = clamp(requestedEnd, start + 1, timelineDuration()); item.text = $('#editText').value.trim() || item.text; item.start_ms = start; item.end_ms = end; updateWordChildren(item, oldStart, oldEnd, start, end, (item.phonemes || []).map(phone => ({start_ms:ms(phone.start_ms),end_ms:ms(phone.end_ms)})), 'resize'); }
  markDraftDirty(); renderSongs();
}

function deleteItem() {
  const annotation = state.selected?.draft?.annotation; if (!annotation) return;
  if (state.selectedPhone !== null) { const [wi,pi] = state.selectedPhone.split(':').map(Number); annotation.words[wi]?.phonemes?.splice(pi,1); } else if (state.selectedWord !== null) annotation.words.splice(state.selectedWord,1);
  state.selectedPhone = null; state.selectedWord = null; markDraftDirty(); renderSongs();
}

function addWord() {
  if (!state.selected?.draft) return; const start = clamp(state.audio ? state.audio.currentTime * 1000 : 0, 0, timelineDuration() - 1);
  const word = {id:`word_local_${Date.now()}`,text:'new word',pronunciation_text:'new word',start_ms:start,end_ms:Math.min(timelineDuration(),start+500),kind:'sung',origin:'manual_corrected',phonemes:[],tags:[]};
  state.selected.draft.annotation.words.push(word); state.selected.draft.annotation.words.sort((a,b)=>a.start_ms-b.start_ms); state.selectedWord = state.selected.draft.annotation.words.indexOf(word); state.selectedPhone = null; markDraftDirty(); renderSongs();
}

function addPhone() {
  const word = wordForSelection(); if (!word) { alert('Select a word first.'); return; }
  const start = clamp(state.audio ? state.audio.currentTime * 1000 : ms(word.start_ms), ms(word.start_ms), ms(word.end_ms) - 1); const phone = {id:`phone_local_${Date.now()}`,symbol:'AA1',start_ms:start,end_ms:Math.min(ms(word.end_ms),start+100),alignment_status:'manual',parent_word_id:word.id,syllable_index:null};
  if (phone.end_ms <= phone.start_ms) { alert('There is no available interval inside this word. Resize the word first.'); return; }
  word.phonemes = [...(word.phonemes || []), phone].sort((a,b)=>a.start_ms-b.start_ms); state.selectedPhone = `${state.selectedWord}:${word.phonemes.indexOf(phone)}`; markDraftDirty(); renderSongs();
}

function zoomAt(factor, event = null) {
  const viewport = $('#timelineViewport'); const before = timelineMetrics(); let anchorTime = state.audio ? state.audio.currentTime * 1000 : 0; let anchorX = viewport ? viewport.clientWidth / 2 : 0;
  if (event && viewport) { const rect = viewport.getBoundingClientRect(); anchorX = event.clientX - rect.left; anchorTime = (viewport.scrollLeft + anchorX) / before.pixelsPerSecond * 1000; }
  state.zoom = clamp(state.zoom * factor, .5, 40); updateTimelineGeometry();
  if (viewport) { const after = timelineMetrics(); viewport.scrollLeft = Math.max(0, anchorTime / 1000 * after.pixelsPerSecond - anchorX); }
}

function wireEditor() {
  const root = $('.editor'); if (!root || !state.selected) return;
  loadAudio(); loadWaveform(); updateTimelineGeometry(); renderInspector();
  root.querySelectorAll('.source-toggle button').forEach(button => button.addEventListener('click', () => { state.kind = button.dataset.kind; state.waveform = null; state.waveformKind = null; renderSongs(); }));
  root.querySelectorAll('.annotation').forEach(button => button.addEventListener('pointerdown', event => beginDrag(event, button, event.target.closest('.annotation-handle')?.dataset.edge || null)));
  root.querySelectorAll('.annotation').forEach(button => button.addEventListener('click', event => { if (state.justDragged) { event.preventDefault(); state.justDragged = false; } }));
  $('#wave').addEventListener('click', event => { const content = $('#timelineContent'); const pps = Number(content.dataset.pixelsPerSecond || 1); const rect = content.getBoundingClientRect(); if (state.audio) state.audio.currentTime = clamp((event.clientX - rect.left) / pps, 0, timelineDuration() / 1000); });
  $('#timelineViewport').addEventListener('wheel', event => {
    // Keep horizontal trackpad and shift-wheel gestures available for the
    // scroll container. Pinch zooms are reported with ctrlKey by browsers.
    const horizontalScroll = event.shiftKey || Math.abs(event.deltaX) > Math.abs(event.deltaY);
    if (horizontalScroll && !event.ctrlKey) return;
    event.preventDefault();
    zoomAt(event.deltaY < 0 ? 1.15 : 1 / 1.15, event);
  }, {passive:false});
  $('#zoomIn').addEventListener('click', () => zoomAt(1.25)); $('#zoomOut').addEventListener('click', () => zoomAt(1 / 1.25)); $('#zoomFit').addEventListener('click', () => { state.zoom = 1; updateTimelineGeometry(); });
  $('#saveDraft').addEventListener('click', saveDraft); $('#approveRevision').addEventListener('click', approve); $('#addWord').addEventListener('click', addWord); $('#addPhone').addEventListener('click', addPhone); $('#deleteSelected').addEventListener('click', deleteItem); $('#removeWordRow').addEventListener('click', () => { if (state.selectedWord !== null) { state.selectedPhone = null; deleteItem(); } }); $('#removePhoneRow').addEventListener('click', () => { if (state.selectedPhone !== null) deleteItem(); }); $('#exportSong').addEventListener('click', exportSong); $('#editLyrics').addEventListener('click', openLyricsDialog);
}

async function saveDraft() { try { await api(`/api/dataset/songs/${state.selected.id}/draft`, {method:'PATCH',body:JSON.stringify(state.selected.draft.annotation)}); await openSong(state.selected.id); } catch (error) { alert(error.message); } }
async function approve() { try { await saveDraft(); await api(`/api/dataset/songs/${state.selected.id}/approve`, {method:'POST'}); await refresh(); await openSong(state.selected.id); } catch (error) { alert(error.message); } }
async function exportSong() { try { const result = await api('/api/dataset/exports', {method:'POST',body:JSON.stringify({song_ids:[state.selected.id]})}); window.open(`/api/dataset/exports/${result.id}/manifest`, '_blank'); } catch (error) { alert(error.message); } }

function openLyricsDialog() { if (!state.selected) return; $('#lyricsInput').value = state.selected.lyrics || ''; $('#lyricsDialog').showModal(); }
async function submitLyrics(event) { event.preventDefault(); if (!state.selected) return; try { const id = state.selected.id; const lyrics = $('#lyricsInput').value; await api(`/api/dataset/songs/${id}/lyrics`, {method:'PATCH',body:JSON.stringify({lyrics})}); await api(`/api/dataset/songs/${id}/extract`, {method:'POST'}); $('#lyricsDialog').close(); await openSong(id); startPolling(); } catch (error) { alert(error.message); } }

window.addEventListener('pointermove', handleDragMove); window.addEventListener('pointerup', endDrag); window.addEventListener('pointercancel', endDrag); window.addEventListener('resize', updateTimelineGeometry);
$('#addSong').addEventListener('click', () => $('#addDialog').showModal()); $('#closeDialog').addEventListener('click', () => $('#addDialog').close()); $('#cancelAdd').addEventListener('click', () => $('#addDialog').close()); $('#audioFile').addEventListener('change', event => { $('#fileName').textContent = event.target.files[0]?.name || 'Choose an audio file'; }); $('#search').addEventListener('input', refresh); $('#statusFilter').addEventListener('change', refresh);
$('#closeLyricsDialog').addEventListener('click', () => $('#lyricsDialog').close()); $('#cancelLyrics').addEventListener('click', () => $('#lyricsDialog').close()); $('#lyricsForm').addEventListener('submit', submitLyrics);
$('#addForm').addEventListener('submit', async event => { event.preventDefault(); const formElement = event.currentTarget; const form = new FormData(formElement); try { const file = $('#audioFile').files[0]; if (!file) throw new Error('Choose an audio file.'); form.set('file', file); const response = await fetch('/api/dataset/songs', {method:'POST',body:form}); if (!response.ok) { const body = await response.json(); throw new Error(body.detail || 'Import failed.'); } const song = await response.json(); formElement.reset(); $('#fileName').textContent = 'Choose an audio file'; $('#addDialog').close(); await refresh(); await openSong(song.id); } catch (error) { alert(error.message); } });
$('#exportAll').addEventListener('click', async () => { try { const result = await api('/api/dataset/exports', {method:'POST',body:'{}'}); window.open(`/api/dataset/exports/${result.id}/manifest`, '_blank'); } catch (error) { alert(error.message); } });
window.addEventListener('load', () => refresh().catch(error => { $('#songList').innerHTML = `<div class="empty error">${escapeHtml(error.message)}</div>`; }));
