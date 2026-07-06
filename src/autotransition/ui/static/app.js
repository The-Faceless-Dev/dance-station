const state = {
  presets: [],
  models: [],
  selectedPreset: null,
  selectedModel: null,
  sourceProbe: null,
  toastTimer: null,
  generationPollTimer: null,
  runtimeRecoveryPollTimer: null,
  isGenerating: false,
  runtimeStatus: null,
  runtimeActivity: null,
  advancedDirty: false,
  generatedResults: [],
  musicResults: [],
  soundEffectResults: [],
  vocal2bgmResults: [],
  voiceWorkStatus: null,
  voiceVoices: [],
  voiceGenerations: [],
  selectedVoiceWorkVoiceId: null,
  voiceWorkSessionStartedAtEpoch: Date.now(),
  voiceWorkRuntimePollTimer: null,
  voiceWorkDependencyPollTimer: null,
  lokrDatasets: [],
  datasetSources: [],
  activeLokrDatasetId: null,
  datasetEditorTargetId: null,
  datasetEditorDonorSourceId: null,
  datasetEditorDonor: null,
  lokrRuns: [],
  lokrAdapters: [],
  selectedLokrRunId: null,
  lokrRunViewClearedAt: Number(window.localStorage.getItem("danceStationLokrRunViewClearedAt") || 0),
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
  rhythmBeatProjects: [],
  rhythmBeatVolumes: [],
  activeRhythmProjectId: null,
  activeRhythmProject: null,
  activeRhythmPlaybackRef: "source",
  selectedRhythmAnalysisId: null,
  visibleRhythmAnalysisIds: [],
  selectedRhythmMergeId: null,
  selectedRhythmSelectionIds: [],
  rhythmSelectionDrafts: {},
  selectedRhythmDraftSegmentIndices: {},
  selectedRhythmSavedSegmentIndices: {},
  rhythmSelectionPointer: null,
  rhythmAnalysisCache: new Map(),
  extractionTracks: [],
  extractionResults: [],
  editorAssets: [],
  localLibraryItems: [],
  localLibraryIndexPath: "",
  publicLibraryConnection: null,
  publicLibraryItems: [],
  voiceWorkTargetInputMode: "upload",
  voiceWorkSourceInputMode: "upload",
  vocal2bgmSourceInputMode: "upload",
  vocal2bgmSourcePath: "",
  vocal2bgmSourceProbe: null,
  vocal2bgmPrompt: "",
  libraryPublishingItemIds: new Set(),
  libraryRevokingItemIds: new Set(),
  selectedEditorAsset: null,
  extractSourceProbe: null,
};

const LIBRARY_WALLET_STORAGE_KEY = "danceStationLibraryWallet";

const el = {
  transitionTabButton: document.querySelector("#transitionTabButton"),
  extractionTabButton: document.querySelector("#extractionTabButton"),
  musicTabButton: document.querySelector("#musicTabButton"),
  voiceWorkTabButton: document.querySelector("#voiceWorkTabButton"),
  datasetEditorTabButton: document.querySelector("#datasetEditorTabButton"),
  lokrTrainingTabButton: document.querySelector("#lokrTrainingTabButton"),
  instrumentLabTabButton: document.querySelector("#instrumentLabTabButton"),
  audioEditTabButton: document.querySelector("#audioEditTabButton"),
  rhythmBeatTabButton: document.querySelector("#rhythmBeatTabButton"),
  libraryTabButton: document.querySelector("#libraryTabButton"),
  transitionPage: document.querySelector("#transitionPage"),
  extractionPage: document.querySelector("#extractionPage"),
  musicPage: document.querySelector("#musicPage"),
  voiceWorkPage: document.querySelector("#voiceWorkPage"),
  datasetEditorPage: document.querySelector("#datasetEditorPage"),
  lokrTrainingPage: document.querySelector("#lokrTrainingPage"),
  instrumentLabPage: document.querySelector("#instrumentLabPage"),
  audioEditPage: document.querySelector("#audioEditPage"),
  rhythmBeatPage: document.querySelector("#rhythmBeatPage"),
  libraryPage: document.querySelector("#libraryPage"),
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
  musicLokrAdapterSelect: document.querySelector("#musicLokrAdapterSelect"),
  musicLokrScale: document.querySelector("#musicLokrScale"),
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
  soundEffectActionState: document.querySelector("#soundEffectActionState"),
  soundEffectLabelInput: document.querySelector("#soundEffectLabelInput"),
  soundEffectPromptInput: document.querySelector("#soundEffectPromptInput"),
  soundEffectDurationInput: document.querySelector("#soundEffectDurationInput"),
  soundEffectStepsInput: document.querySelector("#soundEffectStepsInput"),
  soundEffectOutputFormat: document.querySelector("#soundEffectOutputFormat"),
  runSoundEffectButton: document.querySelector("#runSoundEffectButton"),
  refreshSoundEffectButton: document.querySelector("#refreshSoundEffectButton"),
  soundEffectActivity: document.querySelector("#soundEffectActivity"),
  soundEffectList: document.querySelector("#soundEffectList"),
  vocal2bgmActionState: document.querySelector("#vocal2bgmActionState"),
  vocal2bgmLabelInput: document.querySelector("#vocal2bgmLabelInput"),
  vocal2bgmPromptInput: document.querySelector("#vocal2bgmPromptInput"),
  vocal2bgmSourceUploadMode: document.querySelector("#vocal2bgmSourceUploadMode"),
  vocal2bgmSourceAssetMode: document.querySelector("#vocal2bgmSourceAssetMode"),
  vocal2bgmSourceUploadBlock: document.querySelector("#vocal2bgmSourceUploadBlock"),
  vocal2bgmSourceAssetBlock: document.querySelector("#vocal2bgmSourceAssetBlock"),
  vocal2bgmSourceFile: document.querySelector("#vocal2bgmSourceFile"),
  vocal2bgmSourceFileName: document.querySelector("#vocal2bgmSourceFileName"),
  vocal2bgmSourceAssetSelect: document.querySelector("#vocal2bgmSourceAssetSelect"),
  vocal2bgmSourceAssetName: document.querySelector("#vocal2bgmSourceAssetName"),
  vocal2bgmSourceReadout: document.querySelector("#vocal2bgmSourceReadout"),
  vocal2bgmOutputFormat: document.querySelector("#vocal2bgmOutputFormat"),
  vocal2bgmInferenceSteps: document.querySelector("#vocal2bgmInferenceSteps"),
  vocal2bgmGuidanceScale: document.querySelector("#vocal2bgmGuidanceScale"),
  vocal2bgmShift: document.querySelector("#vocal2bgmShift"),
  vocal2bgmSourceStrength: document.querySelector("#vocal2bgmSourceStrength"),
  vocal2bgmInferMethod: document.querySelector("#vocal2bgmInferMethod"),
  vocal2bgmSeed: document.querySelector("#vocal2bgmSeed"),
  vocal2bgmUseTiledDecode: document.querySelector("#vocal2bgmUseTiledDecode"),
  vocal2bgmDcwEnabled: document.querySelector("#vocal2bgmDcwEnabled"),
  runVocal2BgmButton: document.querySelector("#runVocal2BgmButton"),
  refreshVocal2BgmButton: document.querySelector("#refreshVocal2BgmButton"),
  vocal2bgmActivity: document.querySelector("#vocal2bgmActivity"),
  vocal2bgmList: document.querySelector("#vocal2bgmList"),
  voiceWorkState: document.querySelector("#voiceWorkState"),
  voiceWorkRuntimeActionState: document.querySelector("#voiceWorkRuntimeActionState"),
  installVoiceWorkRuntimeButton: document.querySelector("#installVoiceWorkRuntimeButton"),
  startVoiceWorkRuntimeButton: document.querySelector("#startVoiceWorkRuntimeButton"),
  stopVoiceWorkRuntimeButton: document.querySelector("#stopVoiceWorkRuntimeButton"),
  voiceWorkRuntimeActionDetails: document.querySelector("#voiceWorkRuntimeActionDetails"),
  voiceWorkLabel: document.querySelector("#voiceWorkLabel"),
  voiceWorkLanguage: document.querySelector("#voiceWorkLanguage"),
  voiceWorkDescription: document.querySelector("#voiceWorkDescription"),
  updateVoiceWorkButton: document.querySelector("#updateVoiceWorkButton"),
  refreshVoiceWorkButton: document.querySelector("#refreshVoiceWorkButton"),
  voiceWorkList: document.querySelector("#voiceWorkList"),
  voiceWorkReferenceFiles: document.querySelector("#voiceWorkReferenceFiles"),
  voiceWorkReferenceFilesName: document.querySelector("#voiceWorkReferenceFilesName"),
  voiceWorkAssetSelect: document.querySelector("#voiceWorkAssetSelect"),
  voiceWorkAssetName: document.querySelector("#voiceWorkAssetName"),
  voiceWorkImportAssetButton: document.querySelector("#voiceWorkImportAssetButton"),
  voiceWorkGenerateState: document.querySelector("#voiceWorkGenerateState"),
  voiceWorkSelectedVoice: document.querySelector("#voiceWorkSelectedVoice"),
  voiceWorkRuntimeState: document.querySelector("#voiceWorkRuntimeState"),
  voiceWorkRuntimeDetails: document.querySelector("#voiceWorkRuntimeDetails"),
  voiceWorkTrainingState: document.querySelector("#voiceWorkTrainingState"),
  voiceWorkTargetUploadMode: document.querySelector("#voiceWorkTargetUploadMode"),
  voiceWorkTargetAssetMode: document.querySelector("#voiceWorkTargetAssetMode"),
  voiceWorkTargetUploadBlock: document.querySelector("#voiceWorkTargetUploadBlock"),
  voiceWorkTargetAssetBlock: document.querySelector("#voiceWorkTargetAssetBlock"),
  voiceWorkSampleVoiceSelect: document.querySelector("#voiceWorkSampleVoiceSelect"),
  voiceWorkSampleFile: document.querySelector("#voiceWorkSampleFile"),
  voiceWorkSampleFileName: document.querySelector("#voiceWorkSampleFileName"),
  voiceWorkSampleMode: document.querySelector("#voiceWorkSampleMode"),
  voiceWorkSourceUploadMode: document.querySelector("#voiceWorkSourceUploadMode"),
  voiceWorkSourceAssetMode: document.querySelector("#voiceWorkSourceAssetMode"),
  voiceWorkSourceUploadBlock: document.querySelector("#voiceWorkSourceUploadBlock"),
  voiceWorkSourceAssetBlock: document.querySelector("#voiceWorkSourceAssetBlock"),
  voiceWorkSourceAssetSelect: document.querySelector("#voiceWorkSourceAssetSelect"),
  voiceWorkSourceAssetName: document.querySelector("#voiceWorkSourceAssetName"),
  voiceWorkLoadSourceAssetButton: document.querySelector("#voiceWorkLoadSourceAssetButton"),
  voiceWorkSampleLabel: document.querySelector("#voiceWorkSampleLabel"),
  voiceWorkSampleDiffusionSteps: document.querySelector("#voiceWorkSampleDiffusionSteps"),
  voiceWorkSampleLengthAdjust: document.querySelector("#voiceWorkSampleLengthAdjust"),
  voiceWorkSampleCfgRate: document.querySelector("#voiceWorkSampleCfgRate"),
  convertVoiceWorkSampleButton: document.querySelector("#convertVoiceWorkSampleButton"),
  voiceWorkGenerationList: document.querySelector("#voiceWorkGenerationList"),
  datasetEditorTargetState: document.querySelector("#datasetEditorTargetState"),
  datasetEditorNewLabel: document.querySelector("#datasetEditorNewLabel"),
  datasetEditorCreateButton: document.querySelector("#datasetEditorCreateButton"),
  datasetEditorRefreshButton: document.querySelector("#datasetEditorRefreshButton"),
  datasetEditorTargetList: document.querySelector("#datasetEditorTargetList"),
  datasetEditorTargetReadout: document.querySelector("#datasetEditorTargetReadout"),
  datasetEditorSaveButton: document.querySelector("#datasetEditorSaveButton"),
  datasetEditorLabel: document.querySelector("#datasetEditorLabel"),
  datasetEditorCustomTag: document.querySelector("#datasetEditorCustomTag"),
  datasetEditorDefaultGenre: document.querySelector("#datasetEditorDefaultGenre"),
  datasetEditorDefaultLanguage: document.querySelector("#datasetEditorDefaultLanguage"),
  datasetEditorTagPosition: document.querySelector("#datasetEditorTagPosition"),
  datasetEditorGenreRatio: document.querySelector("#datasetEditorGenreRatio"),
  datasetEditorSampleCount: document.querySelector("#datasetEditorSampleCount"),
  datasetEditorAllInstrumental: document.querySelector("#datasetEditorAllInstrumental"),
  datasetEditorValidationState: document.querySelector("#datasetEditorValidationState"),
  datasetEditorSummary: document.querySelector("#datasetEditorSummary"),
  datasetEditorEntryList: document.querySelector("#datasetEditorEntryList"),
  datasetEditorDonorState: document.querySelector("#datasetEditorDonorState"),
  datasetEditorDonorList: document.querySelector("#datasetEditorDonorList"),
  datasetEditorDonorReadout: document.querySelector("#datasetEditorDonorReadout"),
  datasetEditorDonorEntryList: document.querySelector("#datasetEditorDonorEntryList"),
  datasetEditorJsonFile: document.querySelector("#datasetEditorJsonFile"),
  datasetEditorJsonFileName: document.querySelector("#datasetEditorJsonFileName"),
  datasetEditorCreateFromJsonButton: document.querySelector("#datasetEditorCreateFromJsonButton"),
  datasetEditorAppendJsonButton: document.querySelector("#datasetEditorAppendJsonButton"),
  lokrDatasetState: document.querySelector("#lokrDatasetState"),
  lokrNewDatasetLabel: document.querySelector("#lokrNewDatasetLabel"),
  createLokrDatasetButton: document.querySelector("#createLokrDatasetButton"),
  refreshLokrDatasetsButton: document.querySelector("#refreshLokrDatasetsButton"),
  lokrDatasetJsonFile: document.querySelector("#lokrDatasetJsonFile"),
  lokrDatasetJsonFileName: document.querySelector("#lokrDatasetJsonFileName"),
  createLokrDatasetFromJsonButton: document.querySelector("#createLokrDatasetFromJsonButton"),
  appendLokrDatasetJsonButton: document.querySelector("#appendLokrDatasetJsonButton"),
  lokrDatasetList: document.querySelector("#lokrDatasetList"),
  lokrActiveDatasetReadout: document.querySelector("#lokrActiveDatasetReadout"),
  saveLokrDatasetButton: document.querySelector("#saveLokrDatasetButton"),
  lokrDatasetLabel: document.querySelector("#lokrDatasetLabel"),
  lokrCustomTag: document.querySelector("#lokrCustomTag"),
  lokrDefaultGenre: document.querySelector("#lokrDefaultGenre"),
  lokrDefaultLanguage: document.querySelector("#lokrDefaultLanguage"),
  lokrTagPosition: document.querySelector("#lokrTagPosition"),
  lokrGenreRatio: document.querySelector("#lokrGenreRatio"),
  lokrSampleCount: document.querySelector("#lokrSampleCount"),
  lokrAllInstrumental: document.querySelector("#lokrAllInstrumental"),
  lokrEntryList: document.querySelector("#lokrEntryList"),
  lokrEntryState: document.querySelector("#lokrEntryState"),
  lokrDropZone: document.querySelector("#lokrDropZone"),
  lokrAudioFiles: document.querySelector("#lokrAudioFiles"),
  lokrSelectedFiles: document.querySelector("#lokrSelectedFiles"),
  lokrAssetSelect: document.querySelector("#lokrAssetSelect"),
  addLokrAssetButton: document.querySelector("#addLokrAssetButton"),
  addEmptyLokrEntryButton: document.querySelector("#addEmptyLokrEntryButton"),
  lokrValidationState: document.querySelector("#lokrValidationState"),
  lokrDatasetSummary: document.querySelector("#lokrDatasetSummary"),
  lokrRunState: document.querySelector("#lokrRunState"),
  lokrTrainingReadiness: document.querySelector("#lokrTrainingReadiness"),
  lokrTrainModel: document.querySelector("#lokrTrainModel"),
  lokrTrainEpochs: document.querySelector("#lokrTrainEpochs"),
  lokrTrainSaveEvery: document.querySelector("#lokrTrainSaveEvery"),
  lokrTrainDim: document.querySelector("#lokrTrainDim"),
  lokrTrainAlpha: document.querySelector("#lokrTrainAlpha"),
  lokrTrainOptimizer: document.querySelector("#lokrTrainOptimizer"),
  lokrTrainBatchSize: document.querySelector("#lokrTrainBatchSize"),
  lokrTrainGradAccum: document.querySelector("#lokrTrainGradAccum"),
  lokrTrainChunkDuration: document.querySelector("#lokrTrainChunkDuration"),
  lokrSidestepCommand: document.querySelector("#lokrSidestepCommand"),
  lokrCheckpointDir: document.querySelector("#lokrCheckpointDir"),
  lokrGradientCheckpointing: document.querySelector("#lokrGradientCheckpointing"),
  lokrOffloadEncoder: document.querySelector("#lokrOffloadEncoder"),
  preprocessLokrButton: document.querySelector("#preprocessLokrButton"),
  trainLokrButton: document.querySelector("#trainLokrButton"),
  stopLokrRunButton: document.querySelector("#stopLokrRunButton"),
  clearLokrLogButton: document.querySelector("#clearLokrLogButton"),
  lokrRunList: document.querySelector("#lokrRunList"),
  lokrRunLog: document.querySelector("#lokrRunLog"),
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
  rhythmProjectReadout: document.querySelector("#rhythmProjectReadout"),
  rhythmProjectState: document.querySelector("#rhythmProjectState"),
  rhythmProjectLabel: document.querySelector("#rhythmProjectLabel"),
  createRhythmProjectButton: document.querySelector("#createRhythmProjectButton"),
  refreshRhythmProjectsButton: document.querySelector("#refreshRhythmProjectsButton"),
  rhythmProjectList: document.querySelector("#rhythmProjectList"),
  rhythmSourceState: document.querySelector("#rhythmSourceState"),
  rhythmSourceFile: document.querySelector("#rhythmSourceFile"),
  rhythmSourceFileName: document.querySelector("#rhythmSourceFileName"),
  rhythmSourceAssetSelect: document.querySelector("#rhythmSourceAssetSelect"),
  loadRhythmSourceAssetButton: document.querySelector("#loadRhythmSourceAssetButton"),
  uploadRhythmSourceButton: document.querySelector("#uploadRhythmSourceButton"),
  rhythmSourceSummary: document.querySelector("#rhythmSourceSummary"),
  rhythmPlaybackList: document.querySelector("#rhythmPlaybackList"),
  rhythmSourceAudio: document.querySelector("#rhythmSourceAudio"),
  rhythmTrackState: document.querySelector("#rhythmTrackState"),
  rhythmTrackAssetSelect: document.querySelector("#rhythmTrackAssetSelect"),
  addRhythmTrackButton: document.querySelector("#addRhythmTrackButton"),
  rhythmChartReadout: document.querySelector("#rhythmChartReadout"),
  rhythmSelectionState: document.querySelector("#rhythmSelectionState"),
  rhythmViewMode: document.querySelector("#rhythmViewMode"),
  rhythmActiveAnalysisSelect: document.querySelector("#rhythmActiveAnalysisSelect"),
  rhythmCursorReadout: document.querySelector("#rhythmCursorReadout"),
  rhythmRangeReadout: document.querySelector("#rhythmRangeReadout"),
  rhythmChartScroller: document.querySelector("#rhythmChartScroller"),
  rhythmChartStack: document.querySelector("#rhythmChartStack"),
  rhythmAnalysisState: document.querySelector("#rhythmAnalysisState"),
  rhythmExtractTrackName: document.querySelector("#rhythmExtractTrackName"),
  rhythmExtractTrackLabel: document.querySelector("#rhythmExtractTrackLabel"),
  rhythmExtractGuidanceScale: document.querySelector("#rhythmExtractGuidanceScale"),
  runRhythmTrackExtractionButton: document.querySelector("#runRhythmTrackExtractionButton"),
  rhythmAnalysisTarget: document.querySelector("#rhythmAnalysisTarget"),
  rhythmAnalysisLabel: document.querySelector("#rhythmAnalysisLabel"),
  rhythmWindowSize: document.querySelector("#rhythmWindowSize"),
  rhythmHopSize: document.querySelector("#rhythmHopSize"),
  rhythmSmoothingAlpha: document.querySelector("#rhythmSmoothingAlpha"),
  rhythmMinStrength: document.querySelector("#rhythmMinStrength"),
  rhythmMinProminence: document.querySelector("#rhythmMinProminence"),
  rhythmMinDistanceSeconds: document.querySelector("#rhythmMinDistanceSeconds"),
  runRhythmAnalysisButton: document.querySelector("#runRhythmAnalysisButton"),
  rhythmAnalysisList: document.querySelector("#rhythmAnalysisList"),
  rhythmMergeState: document.querySelector("#rhythmMergeState"),
  rhythmSelectionLabel: document.querySelector("#rhythmSelectionLabel"),
  saveRhythmSelectionButton: document.querySelector("#saveRhythmSelectionButton"),
  rhythmSelectionList: document.querySelector("#rhythmSelectionList"),
  rhythmMergeLabel: document.querySelector("#rhythmMergeLabel"),
  mergeRhythmSelectionsButton: document.querySelector("#mergeRhythmSelectionsButton"),
  finalizeRhythmMergeButton: document.querySelector("#finalizeRhythmMergeButton"),
  rhythmMergeList: document.querySelector("#rhythmMergeList"),
  rhythmAssetState: document.querySelector("#rhythmAssetState"),
  rhythmLyricsModel: document.querySelector("#rhythmLyricsModel"),
  rhythmLyricsLanguage: document.querySelector("#rhythmLyricsLanguage"),
  extractRhythmLyricsButton: document.querySelector("#extractRhythmLyricsButton"),
  rhythmLyricsText: document.querySelector("#rhythmLyricsText"),
  saveRhythmLyricsButton: document.querySelector("#saveRhythmLyricsButton"),
  saveRhythmProjectButton: document.querySelector("#saveRhythmProjectButton"),
  rhythmAssetSummary: document.querySelector("#rhythmAssetSummary"),
  rhythmAssetList: document.querySelector("#rhythmAssetList"),
  rhythmVolumeLabel: document.querySelector("#rhythmVolumeLabel"),
  createRhythmVolumeButton: document.querySelector("#createRhythmVolumeButton"),
  rhythmVolumeList: document.querySelector("#rhythmVolumeList"),
  libraryState: document.querySelector("#libraryState"),
  libraryDetailState: document.querySelector("#libraryDetailState"),
  libraryIndexPath: document.querySelector("#libraryIndexPath"),
  librarySearch: document.querySelector("#librarySearch"),
  libraryKindFilter: document.querySelector("#libraryKindFilter"),
  reindexLibraryButton: document.querySelector("#reindexLibraryButton"),
  refreshLibraryButton: document.querySelector("#refreshLibraryButton"),
  libraryPublishState: document.querySelector("#libraryPublishState"),
  libraryWalletProvider: document.querySelector("#libraryWalletProvider"),
  connectLibraryWalletButton: document.querySelector("#connectLibraryWalletButton"),
  disconnectLibraryWalletButton: document.querySelector("#disconnectLibraryWalletButton"),
  libraryWalletSummary: document.querySelector("#libraryWalletSummary"),
  publicLibraryState: document.querySelector("#publicLibraryState"),
  publicLibraryKind: document.querySelector("#publicLibraryKind"),
  publicLibrarySearch: document.querySelector("#publicLibrarySearch"),
  refreshPublicLibraryButton: document.querySelector("#refreshPublicLibraryButton"),
  publicLibraryList: document.querySelector("#publicLibraryList"),
  libraryList: document.querySelector("#libraryList"),
  librarySummary: document.querySelector("#librarySummary"),
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

function setPreferredLibraryWallet(wallet) {
  window.localStorage.setItem(LIBRARY_WALLET_STORAGE_KEY, wallet);
}

function getPreferredLibraryWallet() {
  const raw = String(window.localStorage.getItem(LIBRARY_WALLET_STORAGE_KEY) || "").trim();
  if (["phantom", "solflare", "backpack", "metamask"].includes(raw)) return raw;
  return "phantom";
}

function getInjectedWalletProviders() {
  const providers = [
    window.phantom && window.phantom.solana,
    window.solflare,
    window.backpack && window.backpack.solana,
    window.solana,
    ...(Array.isArray(window.solana && window.solana.providers) ? window.solana.providers : []),
  ].filter(Boolean);
  return [...new Set(providers)];
}

function getWalletProvider(wallet) {
  const providers = getInjectedWalletProviders();
  if (wallet === "phantom") return providers.find((provider) => provider.isPhantom);
  if (wallet === "solflare") return providers.find((provider) => provider.isSolflare);
  if (wallet === "backpack") return providers.find((provider) => provider.isBackpack);
  if (wallet === "metamask") return providers.find((provider) => provider.isMetaMask);
  return undefined;
}

function walletLabel(connection) {
  const profile = (connection && connection.creator_profile) || {};
  return profile.displayName || connection.public_key || "wallet";
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

function capitalize(value) {
  const text = String(value || "");
  if (!text) return "";
  return `${text.charAt(0).toUpperCase()}${text.slice(1)}`;
}

function voiceRuntimePhaseDisplay(phase) {
  switch (phase) {
    case "missing":
      return { label: "Missing", tone: "error" };
    case "installed":
      return { label: "Installed", tone: "neutral" };
    case "installing":
      return { label: "Installing", tone: "warn" };
    case "starting":
      return { label: "Starting", tone: "warn" };
    case "ready":
      return { label: "Ready", tone: "ok" };
    case "failed":
      return { label: "Failed", tone: "error" };
    case "stale":
      return { label: "Stale", tone: "warn" };
    default:
      return { label: capitalize(phase || "Unknown"), tone: "neutral" };
  }
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
  state.runtimeStatus = runtime;
  const recovery = runtime.recovery || {};
  const recovering = Boolean(recovery.active);
  const tone = recovering ? "warn" : runtime.api_running ? "ok" : runtime.installed ? "warn" : "error";
  const label = recovering
    ? "Recovering"
    : runtime.api_running
      ? "API running"
      : runtime.installed
        ? "Installed"
        : "Not installed";
  setPill(el.runtimeState, label, tone);
  el.runtimeDetails.innerHTML = [
    `<strong>${runtime.message}</strong>`,
    `Install: ${runtime.install_dir}`,
    `API: ${runtime.api_url}`,
    `Managed PID: ${runtime.managed_pid || "none"}${runtime.managed_pid ? ` (${runtime.managed_pid_alive ? "alive" : "not running"})` : ""}`,
    `uv: ${runtime.uv_available ? "available" : "missing"}`,
    `git: ${runtime.git_available ? "available" : "missing"}`,
    `Side-Step: ${runtime.side_step ? runtime.side_step.message : "Not checked"}`,
    `Setup: ${runtime.simple_setup_command}`,
    `Start: ${runtime.simple_start_command}`,
  ].join("<br>");
  if (runtime.side_step_command) {
    el.lokrSidestepCommand.value = runtime.side_step_command;
  }
  el.copyRuntimeCommandButton.dataset.command = `${runtime.simple_setup_command}\n${runtime.simple_start_command}`;
  applyAceRuntimeAvailability();
}

function aceRuntimeBusy() {
  const runtime = state.runtimeStatus || {};
  const recovery = runtime.recovery || {};
  return Boolean(recovery.active);
}

function aceRuntimeReady() {
  return Boolean(state.runtimeStatus && state.runtimeStatus.api_running);
}

function applyAceRuntimeAvailability() {
  const busy = aceRuntimeBusy();
  const ready = aceRuntimeReady();
  const disableAceActions = busy || !ready;
  el.generateButton.disabled = disableAceActions || state.isGenerating;
  el.runExtractionButton.disabled = disableAceActions;
  el.runMusicButton.disabled = disableAceActions;
  if (el.runVocal2BgmButton) el.runVocal2BgmButton.disabled = disableAceActions;
  el.runRhythmTrackExtractionButton.disabled = disableAceActions;
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
  el.voiceWorkPage.classList.toggle("active", page === "voice");
  el.datasetEditorPage.classList.toggle("active", page === "dataseteditor");
  el.lokrTrainingPage.classList.toggle("active", page === "lokr");
  el.instrumentLabPage.classList.toggle("active", page === "instrument");
  el.audioEditPage.classList.toggle("active", page === "audioedit");
  el.rhythmBeatPage.classList.toggle("active", page === "rhythm");
  el.libraryPage.classList.toggle("active", page === "library");
  el.transitionTabButton.classList.toggle("active", page === "transition");
  el.extractionTabButton.classList.toggle("active", page === "extraction");
  el.musicTabButton.classList.toggle("active", page === "music");
  el.voiceWorkTabButton.classList.toggle("active", page === "voice");
  el.datasetEditorTabButton.classList.toggle("active", page === "dataseteditor");
  el.lokrTrainingTabButton.classList.toggle("active", page === "lokr");
  el.instrumentLabTabButton.classList.toggle("active", page === "instrument");
  el.audioEditTabButton.classList.toggle("active", page === "audioedit");
  el.rhythmBeatTabButton.classList.toggle("active", page === "rhythm");
  el.libraryTabButton.classList.toggle("active", page === "library");
  if (page === "instrument") {
    window.setTimeout(drawInstrumentPianoRoll, 50);
  }
  if (page === "rhythm") {
    window.setTimeout(drawRhythmChart, 50);
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

function looksLikePlayableAudio(path) {
  const normalized = String(path || "").toLowerCase();
  return [".mp3", ".wav", ".flac", ".ogg", ".m4a"].some((extension) => normalized.endsWith(extension));
}

function itemAudioFile(item) {
  return (item.files || []).find((file) => String(file.mime_type || "").startsWith("audio/") || looksLikePlayableAudio(file.path));
}

function itemCoverFile(item) {
  return (item.files || []).find((file) => file.role === "cover") || null;
}

function libraryCardImageUrl(item) {
  const coverFile = itemCoverFile(item);
  if (coverFile?.path) {
    return `/api/library/file?path=${encodeURIComponent(coverFile.path)}`;
  }
  const creator = (item.metadata || {}).creator || {};
  return creator.display_image || creator.banner_url || creator.avatar_url || "";
}

function waitForAudioEditorFrame() {
  return new Promise((resolve) => {
    const frame = el.audioEditorFrame;
    if (!frame) {
      resolve(null);
      return;
    }
    const startedAt = Date.now();
    function bridgeWindow() {
      try {
        const frameWindow = frame.contentWindow;
        if (frameWindow && frameWindow.DanceStationAudioMassBridge) return frameWindow;
      } catch (error) {
        return null;
      }
      return null;
    }
    function pollBridge() {
      const frameWindow = bridgeWindow();
      if (frameWindow) {
        frame.removeEventListener("load", onLoad);
        resolve(frameWindow);
        return;
      }
      if (Date.now() - startedAt > 4000) {
        frame.removeEventListener("load", onLoad);
        resolve(frame.contentWindow || null);
        return;
      }
      window.setTimeout(pollBridge, 100);
    }
    const timeout = window.setTimeout(pollBridge, 120);
    const onLoad = () => {
      frame.removeEventListener("load", onLoad);
      window.clearTimeout(timeout);
      pollBridge();
    };
    frame.addEventListener("load", onLoad);
    pollBridge();
  });
}

async function sendAudioBufferToEditor(url, name) {
  const frameWindow = await waitForAudioEditorFrame();
  if (!frameWindow) throw new Error("Audio editor is not ready");
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load audio for editor (${response.status})`);
  const buffer = await response.arrayBuffer();
  frameWindow.postMessage(
    {
      type: "dance-station:load-audio-buffer",
      payload: {
        buffer,
        name,
        mimeType: response.headers.get("content-type") || "audio/wav",
      },
    },
    window.location.origin,
    [buffer],
  );
}

function libraryAudioUrl(item) {
  const audioFile = itemAudioFile(item);
  return audioFile ? `/api/audio?path=${encodeURIComponent(audioFile.path)}` : "";
}

function filteredLibraryItems() {
  const query = el.librarySearch.value.trim().toLowerCase();
  const kind = el.libraryKindFilter.value;
  return state.localLibraryItems.filter((item) => {
    if (kind !== "all" && item.kind !== kind) return false;
    if (!query) return true;
    const filePaths = (item.files || []).map((file) => file.path).join(" ");
    return [item.title, item.kind, (item.tags || []).join(" "), filePaths]
      .join(" ")
      .toLowerCase()
      .includes(query);
  });
}

function filteredPublicLibraryItems() {
  const query = (el.publicLibrarySearch.value || "").trim().toLowerCase();
  if (!query) return state.publicLibraryItems || [];
  return (state.publicLibraryItems || []).filter((item) => {
    const creator = item.creator || {};
    const creatorName = creator.displayName || creator.creatorSlug || "";
    return [
      item.title || "",
      item.kind || "",
      item.description || "",
      creatorName,
      (item.tags || []).join(" "),
    ]
      .join(" ")
      .toLowerCase()
      .includes(query);
  });
}

function activeRhythmProject() {
  return state.activeRhythmProject;
}

function rhythmBeatAssets() {
  return (state.localLibraryItems || []).filter((item) => item.kind === "rhythm_game");
}

function rhythmBeatVolumeOptions() {
  return [...(state.rhythmBeatVolumes || [])].sort((left, right) => Number(left.sort_order || 0) - Number(right.sort_order || 0) || String(left.label || "").localeCompare(String(right.label || "")));
}

function assignableRhythmVolumeOptions(selectedVolumeId = "") {
  return rhythmBeatVolumeOptions().filter((volume) => !volume.official || volume.volume_id === selectedVolumeId);
}

function rhythmBeatVolumeLabel(volumeId) {
  const volume = rhythmBeatVolumeOptions().find((item) => item.volume_id === volumeId);
  return volume ? volume.label : "";
}

function rhythmAssetOptions() {
  return (state.editorAssets || []).filter((asset) => {
    const category = asset.category || "";
    return ["extraction", "merge", "generation", "transition", "edit"].includes(category);
  });
}

function rhythmSourceOptionLabel(item) {
  return `${item.label || item.asset_id} (${item.category || "audio"})`;
}

function renderRhythmAssetSelectors() {
  [el.rhythmSourceAssetSelect, el.rhythmTrackAssetSelect].forEach((select) => {
    if (!select) return;
    const current = select.value;
    select.replaceChildren();
    select.appendChild(option(select === el.rhythmSourceAssetSelect ? "Choose an existing creation" : "Choose an extraction or audio creation", ""));
    rhythmAssetOptions().forEach((asset) => select.appendChild(option(rhythmSourceOptionLabel(asset), asset.asset_id)));
    if (current && rhythmAssetOptions().some((asset) => asset.asset_id === current)) {
      select.value = current;
    }
  });
}

function rhythmProjectAnalysisOptions(project) {
  return project ? (project.analyses || []) : [];
}

function rhythmProjectMergeOptions(project) {
  return project ? (project.merges || []) : [];
}

function renderRhythmProjects() {
  el.rhythmProjectList.replaceChildren();
  setPill(el.rhythmProjectState, state.rhythmBeatProjects.length ? `${state.rhythmBeatProjects.length} projects` : "No projects", state.rhythmBeatProjects.length ? "ok" : "neutral");
  if (!state.rhythmBeatProjects.length) {
    const empty = document.createElement("div");
    empty.className = "empty-result";
    empty.textContent = "No rhythm beat projects yet.";
    el.rhythmProjectList.appendChild(empty);
    return;
  }
  state.rhythmBeatProjects.forEach((project) => {
    const row = document.createElement("article");
    row.className = `generated-item${project.project_id === state.activeRhythmProjectId ? " active" : ""}`;
    row.innerHTML = `
      <div class="generated-title">
        <strong>${escapeHtml(project.label || project.project_id)}</strong>
        <span>${project.final_result_id ? "Final ready" : "Draft"}</span>
      </div>
      <dl class="path-list compact-dl">
        <dt>Source</dt><dd>${escapeHtml(project.source_label || "No source")}</dd>
        <dt>Layers</dt><dd>${Number(project.track_count || 0)} tracks, ${Number(project.selection_count || 0)} selections</dd>
        <dt>Updated</dt><dd>${escapeHtml(formatLibraryDate(project.updated_at || project.created_at))}</dd>
      </dl>
    `;
    row.addEventListener("click", () => loadRhythmProject(project.project_id).catch((error) => showToast(error.message)));
    el.rhythmProjectList.appendChild(row);
  });
}

function rhythmAnalysisTargetOptions(project) {
  const options = [{ value: "source", label: "Source song" }];
  (project.tracks || []).forEach((track) => {
    options.push({ value: `track:${track.track_id}`, label: track.label || track.track_id });
  });
  return options;
}

function syncVisibleRhythmAnalyses(project) {
  const analysisIds = new Set((project?.analyses || []).map((analysis) => analysis.analysis_id));
  state.visibleRhythmAnalysisIds = state.visibleRhythmAnalysisIds.filter((analysisId) => analysisIds.has(analysisId));
  if (!state.visibleRhythmAnalysisIds.length && analysisIds.size) {
    state.visibleRhythmAnalysisIds = [...analysisIds];
  }
}

function rhythmPlaybackEntries(project) {
  if (!project) return [];
  const entries = [];
  if (project.source && project.source.audio_path) {
    entries.push({
      playbackRef: "source",
      kind: "source",
      label: project.source.label || "Source song",
      audioPath: project.source.audio_path,
      durationSeconds: Number(project.source.duration_seconds || 0),
      removable: false,
    });
  }
  (project.tracks || []).forEach((track) => {
    entries.push({
      playbackRef: `track:${track.track_id}`,
      trackId: track.track_id,
      kind: "track",
      label: track.label || track.track_id,
      audioPath: track.audio_path || "",
      durationSeconds: Number(track.duration_seconds || 0),
      removable: true,
      sourceCategory: track.source_category || "",
    });
  });
  return entries;
}

function activeRhythmPlaybackEntry(project = activeRhythmProject()) {
  const entries = rhythmPlaybackEntries(project);
  return entries.find((entry) => entry.playbackRef === state.activeRhythmPlaybackRef) || entries[0] || null;
}

function syncRhythmPlaybackAudio(project = activeRhythmProject()) {
  const entry = activeRhythmPlaybackEntry(project);
  const currentSrc = el.rhythmSourceAudio.getAttribute("data-path") || "";
  const nextSrc = entry?.audioPath || "";
  if (nextSrc !== currentSrc) {
    el.rhythmSourceAudio.pause();
    el.rhythmSourceAudio.removeAttribute("src");
    el.rhythmSourceAudio.removeAttribute("data-path");
    if (nextSrc) {
      el.rhythmSourceAudio.src = `/api/audio?path=${encodeURIComponent(nextSrc)}`;
      el.rhythmSourceAudio.setAttribute("data-path", nextSrc);
      el.rhythmSourceAudio.load();
    }
  }
}

function setRhythmPlaybackSource(playbackRef, options = {}) {
  const project = activeRhythmProject();
  if (!project) return;
  const entries = rhythmPlaybackEntries(project);
  if (!entries.some((entry) => entry.playbackRef === playbackRef)) return;
  const wasPlaying = !el.rhythmSourceAudio.paused;
  state.activeRhythmPlaybackRef = playbackRef;
  syncRhythmPlaybackAudio(project);
  renderRhythmBeatLab();
  if (options.autoplay || (options.preservePlayback && wasPlaying)) {
    el.rhythmSourceAudio.play().catch(() => {});
  }
}

function syncRhythmPlaybackSourceToAnalysis() {
  const project = activeRhythmProject();
  const analysis = currentRhythmAnalysis();
  if (!project || !analysis) return;
  if (analysis.source_type === "track" && analysis.source_ref) {
    setRhythmPlaybackSource(`track:${analysis.source_ref}`);
    return;
  }
  setRhythmPlaybackSource("source");
}

function updateRhythmExtractionLabelPlaceholder() {
  const project = activeRhythmProject();
  const sourceLabel = (project && project.source && project.source.audio_path)
    ? ((project.source.audio_path.split(/[\\/]/).pop() || "source").replace(/\.[^.]+$/, "") || "source")
    : "source";
  const trackType = (el.rhythmExtractTrackName.value || "vocals").trim().toLowerCase() || "vocals";
  el.rhythmExtractTrackLabel.placeholder = `${sourceLabel}_${trackType}`;
}

function renderRhythmPlaybackList(project) {
  el.rhythmPlaybackList.replaceChildren();
  const entries = rhythmPlaybackEntries(project);
  const tracks = (project && project.tracks) || [];
  setPill(el.rhythmTrackState, tracks.length ? `${tracks.length} tracks` : "0 tracks", tracks.length ? "ok" : "neutral");
  if (!entries.length) {
    const empty = document.createElement("div");
    empty.className = "empty-result";
    empty.textContent = "No source or extracted layers yet.";
    el.rhythmPlaybackList.appendChild(empty);
    return;
  }
  entries.forEach((entry) => {
    const row = document.createElement("article");
    row.className = `generated-item${entry.playbackRef === state.activeRhythmPlaybackRef ? " active" : ""}`;
    row.innerHTML = `
      <div class="generated-title">
        <strong>${escapeHtml(entry.label)}</strong>
        <span>${entry.kind === "source" ? "Source" : "Layer"}</span>
      </div>
      <dl class="path-list compact-dl">
        <dt>Duration</dt><dd>${Number(entry.durationSeconds || 0).toFixed(1)}s</dd>
        <dt>Playback</dt><dd>${entry.playbackRef === state.activeRhythmPlaybackRef ? "Active in chart player" : "Available"}</dd>
      </dl>
      <div class="button-row generated-actions">
        <button class="secondary-button rhythm-use-playback-button" type="button">${entry.playbackRef === state.activeRhythmPlaybackRef ? "Loaded" : "Load In Player"}</button>
        ${entry.removable ? '<button class="secondary-button rhythm-remove-track-button" type="button">Remove</button>' : ""}
      </div>
    `;
    row.querySelector(".rhythm-use-playback-button").addEventListener("click", () => {
      setRhythmPlaybackSource(entry.playbackRef, { autoplay: true });
    });
    const removeButton = row.querySelector(".rhythm-remove-track-button");
    if (removeButton && entry.trackId) {
      removeButton.addEventListener("click", () => removeRhythmTrack(entry.trackId));
    }
    el.rhythmPlaybackList.appendChild(row);
  });
}

function renderRhythmAnalysisList(project) {
  el.rhythmAnalysisList.replaceChildren();
  const analyses = rhythmProjectAnalysisOptions(project);
  el.rhythmActiveAnalysisSelect.replaceChildren();
  el.rhythmActiveAnalysisSelect.appendChild(option("Choose analysis", ""));
  analyses.forEach((analysis) => {
    el.rhythmActiveAnalysisSelect.appendChild(option(analysis.label || analysis.analysis_id, analysis.analysis_id));
  });
  if (state.selectedRhythmAnalysisId && analyses.some((analysis) => analysis.analysis_id === state.selectedRhythmAnalysisId)) {
    el.rhythmActiveAnalysisSelect.value = state.selectedRhythmAnalysisId;
  } else if (analyses.length) {
    state.selectedRhythmAnalysisId = analyses[0].analysis_id;
    el.rhythmActiveAnalysisSelect.value = analyses[0].analysis_id;
  }
  if (!analyses.length) {
    const empty = document.createElement("div");
    empty.className = "empty-result";
    empty.textContent = "No saved analyses yet.";
    el.rhythmAnalysisList.appendChild(empty);
    return;
  }
  analyses.forEach((analysis) => {
    const row = document.createElement("article");
    row.className = `generated-item${analysis.analysis_id === state.selectedRhythmAnalysisId ? " active" : ""}`;
    const visibleChecked = state.visibleRhythmAnalysisIds.includes(analysis.analysis_id) ? "checked" : "";
    row.innerHTML = `
      <div class="generated-title">
        <strong>${escapeHtml(analysis.label || analysis.analysis_id)}</strong>
        <span>${escapeHtml(analysis.source_label || analysis.source_ref || "Source")}</span>
      </div>
      <dl class="path-list compact-dl">
        <dt>Beats</dt><dd>${Number((analysis.beat_points || []).length)} peaks</dd>
        <dt>Events</dt><dd>${Number((analysis.source_events || []).length)} events</dd>
      </dl>
      <label class="merge-select"><input class="rhythm-analysis-visible-checkbox" type="checkbox" value="${escapeHtml(analysis.analysis_id)}" ${visibleChecked}/> Show on chart</label>
    `;
    row.addEventListener("click", () => {
      state.selectedRhythmAnalysisId = analysis.analysis_id;
      if (analysis.source_type === "track" && analysis.source_ref) {
        state.activeRhythmPlaybackRef = `track:${analysis.source_ref}`;
      } else {
        state.activeRhythmPlaybackRef = "source";
      }
      renderRhythmBeatLab();
    });
    const visibleCheckbox = row.querySelector(".rhythm-analysis-visible-checkbox");
    visibleCheckbox.addEventListener("click", (event) => {
      event.stopPropagation();
    });
    visibleCheckbox.addEventListener("change", (event) => {
      event.stopPropagation();
      const enabled = event.target.checked;
      if (enabled) {
        state.visibleRhythmAnalysisIds = [...new Set([...state.visibleRhythmAnalysisIds, analysis.analysis_id])];
      } else {
        state.visibleRhythmAnalysisIds = state.visibleRhythmAnalysisIds.filter((id) => id !== analysis.analysis_id);
      }
      renderRhythmBeatLab();
    });
    el.rhythmAnalysisList.appendChild(row);
  });
}

function renderRhythmSelectionList(project) {
  el.rhythmSelectionList.replaceChildren();
  const selections = (project && project.selections) || [];
  if (!selections.length) {
    const empty = document.createElement("div");
    empty.className = "empty-result";
    empty.textContent = "No beat selections yet.";
    el.rhythmSelectionList.appendChild(empty);
    return;
  }
  selections.forEach((selection) => {
    const row = document.createElement("article");
    row.className = "generated-item";
    const checked = state.selectedRhythmSelectionIds.includes(selection.selection_id) ? "checked" : "";
    row.innerHTML = `
      <div class="generated-title">
        <strong>${escapeHtml(selection.label || selection.selection_id)}</strong>
        <span>${escapeHtml(selection.source || "")}</span>
      </div>
      <dl class="path-list compact-dl">
        <dt>Range</dt><dd>${formatTime(selection.range_start_seconds || 0)} to ${formatTime(selection.range_end_seconds || 0)}</dd>
        <dt>Segments</dt><dd>${Number((selection.ranges || []).length || 1)}</dd>
        <dt>Beats</dt><dd>${Number((selection.game_beats || []).length)}</dd>
      </dl>
      <div class="selection-segment-list">
        ${((selection.ranges || []).length ? selection.ranges : [{ start_seconds: selection.range_start_seconds || 0, end_seconds: selection.range_end_seconds || 0 }]).map((range, index) => `
          <div class="selection-segment-item">
            <span>${formatTime(range.start_seconds || 0)} to ${formatTime(range.end_seconds || 0)}</span>
            <button class="secondary-button rhythm-remove-saved-segment-button" type="button" data-segment-index="${index}">Remove Segment</button>
          </div>
        `).join("")}
      </div>
      <label class="merge-select"><input class="rhythm-selection-checkbox" type="checkbox" value="${escapeHtml(selection.selection_id)}" ${checked}/> Select for merge</label>
      <div class="button-row generated-actions">
        <button class="secondary-button rhythm-remove-selection-button" type="button">Remove</button>
      </div>
    `;
    row.querySelector(".rhythm-selection-checkbox").addEventListener("change", (event) => {
      const enabled = event.target.checked;
      if (enabled) {
        state.selectedRhythmSelectionIds = [...new Set([...state.selectedRhythmSelectionIds, selection.selection_id])];
      } else {
        state.selectedRhythmSelectionIds = state.selectedRhythmSelectionIds.filter((id) => id !== selection.selection_id);
      }
      updateRhythmCandidateActionLabel();
    });
    row.querySelector(".rhythm-remove-selection-button").addEventListener("click", () => {
      removeRhythmSelection(selection.selection_id).catch((error) => showToast(error.message));
    });
    row.querySelectorAll(".rhythm-remove-saved-segment-button").forEach((button) => {
      button.addEventListener("click", () => {
        removeSavedRhythmSelectionSegment(selection.selection_id, Number(button.dataset.segmentIndex || 0)).catch((error) => showToast(error.message));
      });
    });
    el.rhythmSelectionList.appendChild(row);
  });
  updateRhythmCandidateActionLabel();
}

function renderRhythmMergeList(project) {
  el.rhythmMergeList.replaceChildren();
  const merges = rhythmProjectMergeOptions(project);
  if (!(state.selectedRhythmMergeId && merges.some((merge) => merge.merge_id === state.selectedRhythmMergeId)) && merges.length) {
    state.selectedRhythmMergeId = merges[0].merge_id;
  }
  if (!merges.length) {
    const empty = document.createElement("div");
    empty.className = "empty-result";
    empty.textContent = "No merge candidates yet.";
    el.rhythmMergeList.appendChild(empty);
    return;
  }
  merges.forEach((merge) => {
    const isSelected = merge.merge_id === state.selectedRhythmMergeId;
    const isFinal = merge.merge_id === project.final_result_id;
    const row = document.createElement("article");
    row.className = `generated-item${isSelected ? " selected" : ""}${isFinal ? " active" : ""}`;
    const status = [isSelected ? "Selected" : null, isFinal ? "Final" : "Candidate"].filter(Boolean).join(" · ");
    row.innerHTML = `
      <div class="generated-title">
        <strong>${escapeHtml(merge.label || merge.merge_id)}</strong>
        <span>${status}</span>
      </div>
      <dl class="path-list compact-dl">
        <dt>Selections</dt><dd>${Number((merge.selection_ids || []).length)}</dd>
        <dt>Beats</dt><dd>${Number((merge.game_beats || []).length)}</dd>
      </dl>
    `;
    row.addEventListener("click", () => {
      state.selectedRhythmMergeId = merge.merge_id;
      renderRhythmBeatLab();
    });
    el.rhythmMergeList.appendChild(row);
  });
}

function renderRhythmAssetList() {
  if (!el.rhythmAssetList) return;
  el.rhythmAssetList.replaceChildren();
  const items = rhythmBeatAssets();
  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "empty-result";
    empty.textContent = "No final rhythm beat assets yet.";
    el.rhythmAssetList.appendChild(empty);
    return;
  }
  items.forEach((item) => {
    const metadata = item.metadata || {};
    const volumeId = String(metadata.volume_id || "");
    const volumeLabel = String(metadata.volume_label || rhythmBeatVolumeLabel(volumeId) || "");
    const supportedModes = metadata.supported_game_modes || {};
    const stepArrowsEnabled = supportedModes.step_arrows !== false;
    const orbBeatEnabled = Boolean(supportedModes.orb_beat);
    const laserShootEnabled = stepArrowsEnabled;
    const modeLabels = [
      stepArrowsEnabled ? "Arrows" : null,
      orbBeatEnabled ? "Orb" : null,
      laserShootEnabled ? "Laser" : null,
    ].filter(Boolean);
    const row = document.createElement("article");
    row.className = "generated-item";
    const publish = metadata.public_library || null;
    row.innerHTML = `
      <div class="generated-title">
        <strong>${escapeHtml(item.title || item.id)}</strong>
        <span>${publish && publish.remote_status === "published" ? "Published" : "Local"}</span>
      </div>
      <dl class="path-list compact-dl">
        <dt>Selections</dt><dd>${escapeHtml(String(metadata.selection_count || 0))}</dd>
        <dt>Merges</dt><dd>${escapeHtml(String(metadata.merge_count || 0))}</dd>
        <dt>Game</dt><dd>${metadata.game_enabled ? "Enabled" : "Hidden"}</dd>
        <dt>Volume</dt><dd>${escapeHtml(volumeLabel || "Unassigned")}</dd>
        <dt>Modes</dt><dd>${escapeHtml(modeLabels.join(", ") || "None")}</dd>
      </dl>
      <div class="control-grid">
        <label class="field">
          <span>Game availability</span>
          <select class="rhythm-asset-enabled" aria-label="Game availability">
            <option value="true"${metadata.game_enabled ? " selected" : ""}>Enabled in games</option>
            <option value="false"${!metadata.game_enabled ? " selected" : ""}>Private to library</option>
          </select>
        </label>
        <label class="field">
          <span>Volume</span>
          <select class="rhythm-asset-volume" aria-label="Rhythm volume">
            <option value="">No volume</option>
            ${assignableRhythmVolumeOptions(volumeId).map((volume) => `<option value="${escapeHtml(volume.volume_id)}"${volume.volume_id === volumeId ? " selected" : ""}>${escapeHtml(volume.label)}</option>`).join("")}
          </select>
        </label>
      </div>
      <div class="toggle-row rhythm-mode-toggle-row">
        <label><input class="rhythm-asset-step-arrows" type="checkbox"${stepArrowsEnabled ? " checked" : ""} /> Step arrows</label>
        <label><input class="rhythm-asset-orb-beat" type="checkbox"${orbBeatEnabled ? " checked" : ""} /> Orb beat</label>
        <span class="mini-state">Laser shoot follows Step arrows</span>
      </div>
    `;
    const updateSettings = () =>
      updateRhythmGameAssetSettings(item.id, {
        game_enabled: row.querySelector(".rhythm-asset-enabled").value === "true",
        volume_id: row.querySelector(".rhythm-asset-volume").value || null,
        step_arrows_enabled: row.querySelector(".rhythm-asset-step-arrows").checked,
        orb_beat_enabled: row.querySelector(".rhythm-asset-orb-beat").checked,
      }).catch((error) => showToast(error.message));
    row.querySelector(".rhythm-asset-enabled").addEventListener("change", updateSettings);
    row.querySelector(".rhythm-asset-volume").addEventListener("change", updateSettings);
    row.querySelector(".rhythm-asset-step-arrows").addEventListener("change", updateSettings);
    row.querySelector(".rhythm-asset-orb-beat").addEventListener("change", updateSettings);
    el.rhythmAssetList.appendChild(row);
  });
}

function renderRhythmVolumeList() {
  if (!el.rhythmVolumeList) return;
  el.rhythmVolumeList.replaceChildren();
  const volumes = rhythmBeatVolumeOptions();
  if (!volumes.length) {
    const empty = document.createElement("div");
    empty.className = "empty-result";
    empty.textContent = "No rhythm-game volumes yet.";
    el.rhythmVolumeList.appendChild(empty);
    return;
  }
  volumes.forEach((volume) => {
    const row = document.createElement("article");
    row.className = "generated-item";
    row.innerHTML = `
      <div class="generated-title">
        <strong>${escapeHtml(volume.label || volume.volume_id)}</strong>
        <span>${volume.official ? "Official" : "Custom"}</span>
      </div>
      <dl class="path-list compact-dl">
        <dt>Slug</dt><dd>${escapeHtml(volume.slug || volume.volume_id)}</dd>
        <dt>Order</dt><dd>${escapeHtml(String(volume.sort_order || 0))}</dd>
      </dl>
      ${volume.official ? "" : `
        <div class="button-row generated-actions">
          <input class="rhythm-volume-rename-input" type="text" value="${escapeHtml(volume.label || "")}" placeholder="Rename volume" />
          <button class="secondary-button rhythm-volume-rename" type="button">Rename Volume</button>
          <button class="secondary-button rhythm-volume-remove" type="button">Remove</button>
        </div>
      `}
    `;
    const renameButton = row.querySelector(".rhythm-volume-rename");
    if (renameButton) {
      renameButton.addEventListener("click", () => {
        const input = row.querySelector(".rhythm-volume-rename-input");
        const label = input ? input.value.trim() : "";
        if (!label) {
          showToast("Enter a volume name");
          return;
        }
        runWithButtonBusyState(renameButton, "Renaming...", async () => {
          const response = await api("/api/rhythm-beats/volumes", {
            method: "POST",
            body: JSON.stringify({ volume_id: volume.volume_id, label }),
          });
          state.rhythmBeatVolumes = response.volumes || [];
          await refreshRhythmProjects();
          await refreshLocalLibrary();
          showToast("Rhythm-game volume renamed");
        }).catch((error) => showToast(error.message));
      });
    }
    const removeButton = row.querySelector(".rhythm-volume-remove");
    if (removeButton) {
      removeButton.addEventListener("click", () => {
        runWithButtonBusyState(removeButton, "Removing...", () => removeRhythmVolume(volume.volume_id)).catch((error) => showToast(error.message));
      });
    }
    el.rhythmVolumeList.appendChild(row);
  });
}

function renderRhythmBeatLab() {
  const project = activeRhythmProject();
  renderRhythmProjects();
  renderRhythmAssetSelectors();
  renderRhythmVolumeList();
  renderRhythmAssetList();
  if (!project) {
    el.rhythmProjectReadout.textContent = "No project selected";
    el.rhythmSourceSummary.textContent = "Attach a song source first. Extracted tracks can then be layered into the project.";
    el.rhythmChartReadout.textContent = "Select a project and run analysis.";
    renderRhythmPlaybackList(null);
    renderRhythmAnalysisList(null);
    renderRhythmSelectionList(null);
    renderRhythmMergeList(null);
    updateRhythmCandidateActionLabel();
    updateRhythmExtractionLabelPlaceholder();
    drawRhythmChart();
    return;
  }
  syncVisibleRhythmAnalyses(project);
  el.rhythmProjectReadout.textContent = `${project.label} · ${project.source.label || "No source"} · ${formatLibraryDate(project.updated_at)}`;
  el.rhythmSourceSummary.textContent = project.source.audio_path
    ? `${project.source.label || "Source"} · ${Number(project.source.duration_seconds || 0).toFixed(1)}s`
    : "Attach a song source first. Extracted tracks can then be layered into the project.";
  if (!state.activeRhythmPlaybackRef || (state.activeRhythmPlaybackRef !== "source" && !(project.tracks || []).some((track) => `track:${track.track_id}` === state.activeRhythmPlaybackRef))) {
    state.activeRhythmPlaybackRef = "source";
  }
  syncRhythmPlaybackAudio(project);
  setPill(el.rhythmSourceState, project.source.audio_path ? "Ready" : "No source", project.source.audio_path ? "ok" : "neutral");
  setPill(el.rhythmAssetState, project.final_result_id ? "Final ready" : "No final", project.final_result_id ? "ok" : "neutral");
  el.rhythmLyricsText.value = (((project.lyrics || {}).text) || "");

  el.rhythmAnalysisTarget.replaceChildren();
  rhythmAnalysisTargetOptions(project).forEach((entry) => el.rhythmAnalysisTarget.appendChild(option(entry.label, entry.value)));
  renderRhythmPlaybackList(project);
  renderRhythmAnalysisList(project);
  renderRhythmSelectionList(project);
  renderRhythmMergeList(project);
  updateRhythmCandidateActionLabel();
  updateRhythmExtractionLabelPlaceholder();
  drawRhythmChart();
}

function updateRhythmCandidateActionLabel() {
  const count = state.selectedRhythmSelectionIds.length;
  if (!el.mergeRhythmSelectionsButton) return;
  if (count > 1) {
    el.mergeRhythmSelectionsButton.textContent = "Merge Selected Layers Into Candidate";
    return;
  }
  el.mergeRhythmSelectionsButton.textContent = "Create Candidate From Selected Layers";
}

async function runWithButtonBusyState(button, busyLabel, task, options = {}) {
  if (!button) {
    return task();
  }
  if (button.dataset.busy === "true") {
    return null;
  }
  const restore = options.restore || (() => {});
  const idleLabel = button.textContent;
  button.dataset.busy = "true";
  button.disabled = true;
  if (busyLabel) {
    button.textContent = busyLabel;
  }
  try {
    return await task();
  } finally {
    button.dataset.busy = "false";
    button.disabled = false;
    button.textContent = idleLabel;
    restore();
  }
}

async function refreshRhythmProjects() {
  state.rhythmBeatProjects = await api("/api/rhythm-beats/projects");
  await refreshRhythmVolumes();
  if (state.activeRhythmProjectId && !state.rhythmBeatProjects.some((project) => project.project_id === state.activeRhythmProjectId)) {
    state.activeRhythmProjectId = null;
    state.activeRhythmProject = null;
  }
  if (!state.activeRhythmProjectId && state.rhythmBeatProjects.length) {
    await loadRhythmProject(state.rhythmBeatProjects[0].project_id, false);
    return;
  }
  renderRhythmBeatLab();
}

async function loadRhythmProject(projectId, refreshList = true) {
  const project = await api(`/api/rhythm-beats/projects/${encodeURIComponent(projectId)}`);
  const previousProjectId = state.activeRhythmProjectId;
  const previousVisibleIds = [...state.visibleRhythmAnalysisIds];
  state.activeRhythmProjectId = project.project_id;
  state.activeRhythmProject = project;
  state.activeRhythmPlaybackRef = "source";
  const analysisIds = (project.analyses || []).map((analysis) => analysis.analysis_id);
  if (!analysisIds.includes(state.selectedRhythmAnalysisId)) {
    state.selectedRhythmAnalysisId = analysisIds[0] || null;
  }
  if (previousProjectId === project.project_id) {
    const validVisibleIds = previousVisibleIds.filter((analysisId) => analysisIds.includes(analysisId));
    state.visibleRhythmAnalysisIds = validVisibleIds.length ? validVisibleIds : [...analysisIds];
  } else {
    state.visibleRhythmAnalysisIds = [...analysisIds];
  }
  state.selectedRhythmMergeId = (project.merges || [])[0]?.merge_id || null;
  state.selectedRhythmSelectionIds = [];
  state.rhythmSelectionDrafts = {};
  state.selectedRhythmDraftSegmentIndices = {};
  state.selectedRhythmSavedSegmentIndices = {};
  state.rhythmSelectionPointer = null;
  if (refreshList) {
    state.rhythmBeatProjects = await api("/api/rhythm-beats/projects");
    await refreshRhythmVolumes();
  }
  renderRhythmBeatLab();
}

async function refreshRhythmVolumes() {
  const volumeResponse = await api("/api/rhythm-beats/volumes");
  state.rhythmBeatVolumes = volumeResponse.volumes || [];
  return state.rhythmBeatVolumes;
}

async function saveRhythmProject(showSavedToast = false) {
  const project = activeRhythmProject();
  if (!project) {
    showToast("Create or load a rhythm beat project");
    return null;
  }
  const response = await api(`/api/rhythm-beats/projects/${encodeURIComponent(project.project_id)}`, {
    method: "PATCH",
    body: JSON.stringify({ project }),
  });
  state.activeRhythmProject = response.project;
  state.activeRhythmProjectId = response.project.project_id;
  state.rhythmBeatProjects = await api("/api/rhythm-beats/projects");
  await refreshLocalLibrary();
  renderRhythmBeatLab();
  if (showSavedToast) showToast("Rhythm beat project saved");
  return response.project;
}

async function createRhythmVolume() {
  const label = (el.rhythmVolumeLabel && el.rhythmVolumeLabel.value.trim()) || "";
  if (!label) {
    showToast("Enter a volume name");
    return;
  }
  const response = await api("/api/rhythm-beats/volumes", {
    method: "POST",
    body: JSON.stringify({ label }),
  });
  state.rhythmBeatVolumes = response.volumes || [];
  if (el.rhythmVolumeLabel) {
    el.rhythmVolumeLabel.value = "";
  }
  renderRhythmBeatLab();
  showToast("Rhythm-game volume saved");
}

async function removeRhythmVolume(volumeId) {
  const response = await api(`/api/rhythm-beats/volumes/${encodeURIComponent(volumeId)}`, {
    method: "DELETE",
  });
  state.rhythmBeatVolumes = response.volumes || [];
  await refreshRhythmProjects();
  showToast("Rhythm-game volume removed");
}

async function updateRhythmGameAssetSettings(projectId, payload) {
  const response = await api(`/api/rhythm-beats/projects/${encodeURIComponent(projectId)}/game-asset`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
  if (response.project && state.activeRhythmProjectId === response.project.project_id) {
    state.activeRhythmProject = response.project;
  }
  if (response.volumes) {
    state.rhythmBeatVolumes = response.volumes;
  }
  await refreshLocalLibrary();
  await refreshRhythmProjects();
  showToast("Rhythm-game asset updated");
}

function currentRhythmAnalysis() {
  const project = activeRhythmProject();
  if (!project) return null;
  return (project.analyses || []).find((analysis) => analysis.analysis_id === state.selectedRhythmAnalysisId) || null;
}

function currentRhythmMerge() {
  const project = activeRhythmProject();
  if (!project) return null;
  return (project.merges || []).find((merge) => merge.merge_id === state.selectedRhythmMergeId) || null;
}

function activeRhythmSelectionDraft() {
  const analysis = currentRhythmAnalysis();
  if (!analysis) return null;
  return state.rhythmSelectionDrafts[analysis.analysis_id] || null;
}

function rhythmDisplayDuration() {
  const project = activeRhythmProject();
  return Number((project && project.source && project.source.duration_seconds) || 0);
}

function rhythmChartMetrics() {
  const duration = Math.max(1, rhythmDisplayDuration());
  const width = Math.max(1400, Math.ceil(duration * 48));
  const height = 118;
  return { duration, width, height, left: 16, right: 16, top: 12, bottom: 16 };
}

function rhythmClientXToTime(clientX) {
  const scroller = el.rhythmChartScroller;
  const rect = scroller.getBoundingClientRect();
  const metrics = rhythmChartMetrics();
  const localX = Math.max(0, Math.min(rect.width, clientX - rect.left));
  const chartX = scroller.scrollLeft + localX;
  return rhythmXToTime(chartX, metrics);
}

function maybeAutoScrollRhythmChart(clientX) {
  const scroller = el.rhythmChartScroller;
  const rect = scroller.getBoundingClientRect();
  const edgeThreshold = 40;
  const scrollStep = 20;
  if (clientX < rect.left + edgeThreshold) {
    scroller.scrollLeft = Math.max(0, scroller.scrollLeft - scrollStep);
    return;
  }
  if (clientX > rect.right - edgeThreshold) {
    scroller.scrollLeft = Math.min(scroller.scrollWidth - scroller.clientWidth, scroller.scrollLeft + scrollStep);
  }
}

function rhythmChartRows(project) {
  const rows = [];
  const analyses = (project?.analyses || []).filter((analysis) => state.visibleRhythmAnalysisIds.includes(analysis.analysis_id));
  analyses.forEach((analysis) => {
    rows.push({
      rowId: `analysis:${analysis.analysis_id}`,
      type: "analysis",
      label: analysis.label || analysis.source_label || "Analysis",
      subtitle: analysis.source_label || analysis.source_ref || "Source",
      analysis,
    });
  });
  (project?.selections || []).forEach((selection) => {
    rows.push({
      rowId: `selection:${selection.selection_id}`,
      type: "selection",
      label: selection.label || selection.selection_id,
      subtitle: selection.source || "",
      selection,
    });
  });
  const selectedMerge = currentRhythmMerge();
  if (selectedMerge) {
    rows.push({
      rowId: `merge:${selectedMerge.merge_id}`,
      type: "merge",
      label: selectedMerge.label || selectedMerge.merge_id,
      subtitle: selectedMerge.merge_id === project?.final_result_id ? "Selected merge · Final candidate" : "Selected merge candidate",
      merge: selectedMerge,
      isFinal: selectedMerge.merge_id === project?.final_result_id,
    });
  }
  const finalMerge = project?.final_result_id
    ? (project?.merges || []).find((merge) => merge.merge_id === project.final_result_id)
    : null;
  if (el.rhythmViewMode.value === "final" && finalMerge && finalMerge.merge_id !== selectedMerge?.merge_id) {
    rows.push({
      rowId: `merge-final:${finalMerge.merge_id}`,
      type: "merge",
      label: finalMerge.label || finalMerge.merge_id,
      subtitle: "Final candidate",
      merge: finalMerge,
      isFinal: true,
    });
  }
  return rows;
}

function rhythmTimeToX(timeSeconds, metrics) {
  const usable = metrics.width - metrics.left - metrics.right;
  return metrics.left + (Math.max(0, Math.min(metrics.duration, timeSeconds)) / metrics.duration) * usable;
}

function rhythmXToTime(x, metrics) {
  const usable = metrics.width - metrics.left - metrics.right;
  const ratio = Math.max(0, Math.min(1, (x - metrics.left) / usable));
  return ratio * metrics.duration;
}

function drawRhythmChart() {
  const project = activeRhythmProject();
  const playbackEntry = activeRhythmPlaybackEntry(project);
  const metrics = rhythmChartMetrics();
  el.rhythmChartStack.style.width = `${metrics.width + 20}px`;
  el.rhythmChartStack.replaceChildren();
  const rows = rhythmChartRows(project);
  const analysis = currentRhythmAnalysis();
  if (!rows.length) {
    const empty = document.createElement("div");
    empty.className = "empty-result";
    empty.textContent = "Run analysis to start beat authoring.";
    el.rhythmChartStack.appendChild(empty);
    return;
  }
  const ns = "http://www.w3.org/2000/svg";
  rows.forEach((row) => {
    const container = document.createElement("section");
    container.className = "rhythm-chart-row";
    container.dataset.rowType = row.type;
    if (row.analysis) container.dataset.analysisId = row.analysis.analysis_id;
    if (row.merge) container.dataset.mergeId = row.merge.merge_id;
    if (row.isFinal) container.dataset.final = "true";

    const header = document.createElement("div");
    header.className = "rhythm-chart-row-header";
    header.innerHTML = `
      <strong>${escapeHtml(row.label)}</strong>
      <span>${escapeHtml(row.subtitle)}</span>
    `;
    container.appendChild(header);

    const actionRail = document.createElement("div");
    actionRail.className = "rhythm-chart-row-actions";
    container.appendChild(actionRail);

    const track = document.createElement("div");
    track.className = "rhythm-chart-row-track";
    track.style.width = `${metrics.width}px`;
    const svg = document.createElementNS(ns, "svg");
    svg.classList.add("rhythm-chart-svg");
    if (row.analysis) svg.dataset.analysisId = row.analysis.analysis_id;
    svg.setAttribute("viewBox", `0 0 ${metrics.width} ${metrics.height}`);
    svg.style.width = `${metrics.width}px`;
    svg.style.height = `${metrics.height}px`;

    for (let second = 0; second <= metrics.duration; second += 1) {
      const line = document.createElementNS(ns, "line");
      const x = rhythmTimeToX(second, metrics);
      line.setAttribute("x1", String(x));
      line.setAttribute("x2", String(x));
      line.setAttribute("y1", String(metrics.top));
      line.setAttribute("y2", String(metrics.height - metrics.bottom));
      line.setAttribute("stroke", second % 4 === 0 ? "#323741" : "rgba(50,55,65,0.45)");
      line.setAttribute("stroke-width", second % 4 === 0 ? "1.5" : "1");
      svg.appendChild(line);
    }

    const baseLine = document.createElementNS(ns, "line");
    baseLine.setAttribute("x1", String(metrics.left));
    baseLine.setAttribute("x2", String(metrics.width - metrics.right));
    baseLine.setAttribute("y1", String(metrics.height - metrics.bottom));
    baseLine.setAttribute("y2", String(metrics.height - metrics.bottom));
    baseLine.setAttribute("stroke", "#323741");
    svg.appendChild(baseLine);

    if (row.type === "analysis" && row.analysis) {
      (row.analysis.beat_points || []).forEach((beat) => {
        const bar = document.createElementNS(ns, "line");
        const x = rhythmTimeToX(Number(beat.timeSeconds || 0), metrics);
        const strength = Math.max(0.12, Number(beat.strength || 0));
        bar.setAttribute("x1", String(x));
        bar.setAttribute("x2", String(x));
        bar.setAttribute("y1", String(metrics.height - metrics.bottom));
        bar.setAttribute("y2", String(metrics.height - metrics.bottom - strength * (metrics.height - metrics.top - metrics.bottom - 8)));
        bar.setAttribute("stroke", "#2dd4bf");
        bar.setAttribute("stroke-width", "2");
        svg.appendChild(bar);
      });
    } else if (row.selection) {
      const savedRanges = row.selection.ranges || [{ start_seconds: row.selection.range_start_seconds || 0, end_seconds: row.selection.range_end_seconds || 0 }];
      const selectedSavedIndex = state.selectedRhythmSavedSegmentIndices[row.selection.selection_id] ?? -1;
      if (selectedSavedIndex >= 0 && selectedSavedIndex < savedRanges.length) {
        const clearButton = document.createElement("button");
        clearButton.type = "button";
        clearButton.className = "secondary-button rhythm-clear-draft-segment-button";
        clearButton.textContent = "Clear Segment";
        clearButton.addEventListener("click", (event) => {
          event.stopPropagation();
          removeSavedRhythmSelectionSegment(row.selection.selection_id, selectedSavedIndex).catch((error) => showToast(error.message));
        });
        actionRail.appendChild(clearButton);
      }
      savedRanges.forEach((range, rangeIndex) => {
        const overlay = document.createElementNS(ns, "rect");
        overlay.setAttribute("x", String(rhythmTimeToX(range.start_seconds || 0, metrics)));
        overlay.setAttribute("y", String(metrics.top));
        overlay.setAttribute("width", String(Math.max(6, rhythmTimeToX(range.end_seconds || 0, metrics) - rhythmTimeToX(range.start_seconds || 0, metrics))));
        overlay.setAttribute("height", String(metrics.height - metrics.top - metrics.bottom));
        overlay.setAttribute("fill", selectedSavedIndex === rangeIndex ? "rgba(242,184,75,0.26)" : "rgba(242,184,75,0.16)");
        overlay.setAttribute("stroke", selectedSavedIndex === rangeIndex ? "#fde68a" : "#f2b84b");
        overlay.setAttribute("stroke-width", selectedSavedIndex === rangeIndex ? "2" : "1");
        overlay.setAttribute("rx", "4");
        overlay.classList.add("rhythm-saved-segment");
        overlay.dataset.selectionId = row.selection.selection_id;
        overlay.dataset.segmentIndex = String(rangeIndex);
        svg.appendChild(overlay);
      });
      (row.selection.game_beats || []).forEach((beat) => {
        const dot = document.createElementNS(ns, "circle");
        dot.setAttribute("cx", String(rhythmTimeToX(Number(beat.timeSeconds || 0), metrics)));
        dot.setAttribute("cy", String((metrics.top + metrics.height - metrics.bottom) / 2));
        dot.setAttribute("r", "3");
        dot.setAttribute("fill", "#f2b84b");
        svg.appendChild(dot);
      });
    } else if (row.type === "merge" && row.merge) {
      (row.merge.game_beats || []).forEach((beat) => {
        const line = document.createElementNS(ns, "line");
        const x = rhythmTimeToX(Number(beat.timeSeconds || 0), metrics);
        const strength = Math.max(0.12, Number(beat.strength || 0));
        line.setAttribute("x1", String(x));
        line.setAttribute("x2", String(x));
        line.setAttribute("y1", String(metrics.height - metrics.bottom));
        line.setAttribute("y2", String(metrics.height - metrics.bottom - strength * (metrics.height - metrics.top - metrics.bottom - 8)));
        line.setAttribute("stroke", row.isFinal ? "#86efac" : "#60a5fa");
        line.setAttribute("stroke-width", row.isFinal ? "2.5" : "2");
        svg.appendChild(line);
      });
    }

    const rowDraft = row.analysis ? state.rhythmSelectionDrafts[row.analysis.analysis_id] : null;
    if (rowDraft && row.analysis) {
      const activeDraftRanges = rowDraft.ranges || [];
      const selectedDraftIndex = state.selectedRhythmDraftSegmentIndices[row.analysis.analysis_id] ?? -1;
      if (selectedDraftIndex >= 0 && selectedDraftIndex < activeDraftRanges.length) {
        const clearButton = document.createElement("button");
        clearButton.type = "button";
        clearButton.className = "secondary-button rhythm-clear-draft-segment-button";
        clearButton.textContent = "Clear Segment";
        clearButton.addEventListener("click", (event) => {
          event.stopPropagation();
          removeRhythmDraftSegment(row.analysis.analysis_id, selectedDraftIndex);
        });
        actionRail.appendChild(clearButton);
      }
      activeDraftRanges.forEach((range) => {
        const rect = document.createElementNS(ns, "rect");
        const start = Math.min(range.startSeconds, range.endSeconds);
        const end = Math.max(range.startSeconds, range.endSeconds);
        const rangeIndex = activeDraftRanges.indexOf(range);
        rect.setAttribute("x", String(rhythmTimeToX(start, metrics)));
        rect.setAttribute("y", String(metrics.top));
        rect.setAttribute("width", String(Math.max(4, rhythmTimeToX(end, metrics) - rhythmTimeToX(start, metrics))));
        rect.setAttribute("height", String(metrics.height - metrics.top - metrics.bottom));
        rect.setAttribute("fill", selectedDraftIndex === rangeIndex ? "rgba(45,212,191,0.22)" : "rgba(45,212,191,0.12)");
        rect.setAttribute("stroke", selectedDraftIndex === rangeIndex ? "#86efac" : "#2dd4bf");
        rect.setAttribute("stroke-width", selectedDraftIndex === rangeIndex ? "2" : "1");
        rect.classList.add("rhythm-draft-segment");
        rect.dataset.analysisId = row.analysis.analysis_id;
        rect.dataset.segmentIndex = String(rangeIndex);
        svg.appendChild(rect);
      });
    }

    const audio = el.rhythmSourceAudio;
    if (audio && Number.isFinite(audio.currentTime)) {
      const cursor = document.createElementNS(ns, "line");
      const x = rhythmTimeToX(Math.min(metrics.duration, audio.currentTime || 0), metrics);
      cursor.setAttribute("x1", String(x));
      cursor.setAttribute("x2", String(x));
      cursor.setAttribute("y1", String(metrics.top));
      cursor.setAttribute("y2", String(metrics.height - metrics.bottom));
      cursor.setAttribute("stroke", "#f87171");
      cursor.setAttribute("stroke-width", "2");
      svg.appendChild(cursor);
      el.rhythmCursorReadout.textContent = formatTime(audio.currentTime || 0);
    }

    track.appendChild(svg);
    container.appendChild(track);
    el.rhythmChartStack.appendChild(container);
  });

  const activeDraft = activeRhythmSelectionDraft();
  if (activeDraft) {
    const ranges = activeDraft.ranges || [];
    const starts = ranges.map((range) => Math.min(range.startSeconds, range.endSeconds));
    const ends = ranges.map((range) => Math.max(range.startSeconds, range.endSeconds));
    if (ranges.length) {
      el.rhythmRangeReadout.textContent = `${ranges.length} segments · ${formatTime(Math.min(...starts))} to ${formatTime(Math.max(...ends))}`;
    } else {
      el.rhythmRangeReadout.textContent = "No selected range";
    }
  } else {
    el.rhythmRangeReadout.textContent = "No selected range";
  }
  if (project) {
    const analysis = currentRhythmAnalysis();
    const playbackLabel = playbackEntry ? playbackEntry.label : "No playback source";
    const analysisLabel = analysis ? (analysis.label || analysis.source_label || "Analysis") : "No analysis";
    const merge = currentRhythmMerge();
    const mergeLabel = merge ? (merge.label || merge.merge_id) : "No candidate selected";
    el.rhythmChartReadout.textContent = `${analysisLabel} · Player: ${playbackLabel} · Candidate: ${mergeLabel}`;
  }
}

async function createRhythmProject() {
  const response = await api("/api/rhythm-beats/projects", {
    method: "POST",
    body: JSON.stringify({ label: el.rhythmProjectLabel.value.trim() || "New rhythm beat project" }),
  });
  state.activeRhythmProject = response.project;
  state.activeRhythmProjectId = response.project.project_id;
  showToast("Rhythm beat project created");
  await refreshRhythmProjects();
}

async function uploadRhythmSource() {
  const project = activeRhythmProject();
  const file = el.rhythmSourceFile.files && el.rhythmSourceFile.files[0];
  if (!project) {
    showToast("Create or load a project first");
    return;
  }
  if (!file) {
    showToast("Choose an audio file");
    return;
  }
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`/api/rhythm-beats/projects/${encodeURIComponent(project.project_id)}/source/upload`, {
    method: "POST",
    body: formData,
  });
  const body = await response.json().catch(() => null);
  if (!response.ok) throw new Error(body && body.detail ? body.detail : `Upload failed: ${response.status}`);
  state.activeRhythmProject = body.project;
  state.activeRhythmPlaybackRef = "source";
  showToast("Rhythm source attached");
  await refreshRhythmProjects();
}

async function attachRhythmSourceAsset() {
  const project = activeRhythmProject();
  const assetId = el.rhythmSourceAssetSelect.value;
  if (!project) {
    showToast("Create or load a project first");
    return;
  }
  if (!assetId) {
    showToast("Choose an existing creation");
    return;
  }
  const response = await api(`/api/rhythm-beats/projects/${encodeURIComponent(project.project_id)}/source/asset`, {
    method: "POST",
    body: JSON.stringify({ asset_id: assetId }),
  });
  state.activeRhythmProject = response.project;
  state.activeRhythmPlaybackRef = "source";
  showToast("Source creation attached");
  await refreshRhythmProjects();
}

async function addRhythmTrack() {
  const project = activeRhythmProject();
  const assetId = el.rhythmTrackAssetSelect.value;
  if (!project) {
    showToast("Create or load a project first");
    return;
  }
  if (!assetId) {
    showToast("Choose an existing creation");
    return;
  }
  const response = await api(`/api/rhythm-beats/projects/${encodeURIComponent(project.project_id)}/tracks/asset`, {
    method: "POST",
    body: JSON.stringify({ asset_id: assetId }),
  });
  state.activeRhythmProject = response.project;
  showToast("Linked track added");
  await refreshRhythmProjects();
}

async function removeRhythmTrack(trackId) {
  const project = activeRhythmProject();
  if (!project) return;
  const response = await api(`/api/rhythm-beats/projects/${encodeURIComponent(project.project_id)}/tracks/${encodeURIComponent(trackId)}`, {
    method: "DELETE",
  });
  state.activeRhythmProject = response.project;
  if (state.activeRhythmPlaybackRef === `track:${trackId}`) {
    state.activeRhythmPlaybackRef = "source";
  }
  showToast("Linked track removed");
  await refreshRhythmProjects();
}

async function removeRhythmSelection(selectionId) {
  const project = activeRhythmProject();
  if (!project) return;
  const response = await api(`/api/rhythm-beats/projects/${encodeURIComponent(project.project_id)}/selections/${encodeURIComponent(selectionId)}`, {
    method: "DELETE",
  });
  state.activeRhythmProject = response.project;
  state.selectedRhythmSelectionIds = state.selectedRhythmSelectionIds.filter((id) => id !== selectionId);
  showToast("Saved selection removed");
  await refreshRhythmProjects();
}

function removeRhythmDraftSegment(analysisId, segmentIndex) {
  const draft = state.rhythmSelectionDrafts[analysisId];
  if (!draft) return;
  draft.ranges = (draft.ranges || []).filter((_, index) => index !== segmentIndex);
  if (!draft.ranges.length) {
    delete state.rhythmSelectionDrafts[analysisId];
    delete state.selectedRhythmDraftSegmentIndices[analysisId];
  } else if ((state.selectedRhythmDraftSegmentIndices[analysisId] ?? -1) >= draft.ranges.length) {
    state.selectedRhythmDraftSegmentIndices[analysisId] = draft.ranges.length - 1;
  }
  drawRhythmChart();
}

function selectRhythmDraftSegment(analysisId, segmentIndex) {
  state.selectedRhythmDraftSegmentIndices[analysisId] = segmentIndex;
  drawRhythmChart();
}

function selectRhythmSavedSegment(selectionId, segmentIndex) {
  state.selectedRhythmSavedSegmentIndices[selectionId] = segmentIndex;
  drawRhythmChart();
}

async function removeSavedRhythmSelectionSegment(selectionId, segmentIndex) {
  const project = activeRhythmProject();
  if (!project) return;
  const selections = [...(project.selections || [])];
  const selectionIndex = selections.findIndex((selection) => selection.selection_id === selectionId);
  if (selectionIndex < 0) return;
  const selection = { ...selections[selectionIndex] };
  const ranges = [...((selection.ranges || []).map((range) => ({ ...range })))];
  if (!ranges.length) return;
  selection.ranges = ranges.filter((_, index) => index !== segmentIndex);
  if (!selection.ranges.length) {
    selections.splice(selectionIndex, 1);
    project.selections = selections;
    delete state.selectedRhythmSavedSegmentIndices[selectionId];
    await saveRhythmProject();
    showToast("Saved selection removed");
    return;
  }
  const analysis = (project.analyses || []).find((item) => item.analysis_id === selection.analysis_id);
  if (analysis) {
    selection.game_beats = (analysis.beat_points || []).filter((beat) => {
      const time = Number(beat.timeSeconds || 0);
      return selection.ranges.some((range) => time >= Number(range.start_seconds || 0) && time <= Number(range.end_seconds || 0));
    });
    selection.game_notes = (analysis.source_events || []).filter((event) => {
      const start = Number(event.startSeconds || 0);
      const end = Number(event.endSeconds || 0);
      return selection.ranges.some((range) => end >= Number(range.start_seconds || 0) && start <= Number(range.end_seconds || 0));
    });
  }
  selection.range_start_seconds = Math.min(...selection.ranges.map((range) => Number(range.start_seconds || 0)));
  selection.range_end_seconds = Math.max(...selection.ranges.map((range) => Number(range.end_seconds || 0)));
  selection.game_beat_selections = (selection.game_beat_selections || []).filter((_, index) => index !== segmentIndex);
  const nextSelectedIndex = state.selectedRhythmSavedSegmentIndices[selectionId] ?? -1;
  if (nextSelectedIndex >= selection.ranges.length) {
    state.selectedRhythmSavedSegmentIndices[selectionId] = selection.ranges.length - 1;
  }
  selections[selectionIndex] = selection;
  project.selections = selections;
  await saveRhythmProject();
  showToast("Selection segment removed");
}

function rhythmAnalysisConfig() {
  return {
    windowSize: numericValue(el.rhythmWindowSize) ?? 2048,
    hopSize: numericValue(el.rhythmHopSize) ?? 512,
    smoothingAlpha: numericValue(el.rhythmSmoothingAlpha) ?? 0.35,
    minStrength: numericValue(el.rhythmMinStrength) ?? 0.16,
    minProminence: numericValue(el.rhythmMinProminence) ?? 0.05,
    minDistanceSeconds: numericValue(el.rhythmMinDistanceSeconds) ?? 0.22,
  };
}

function mixToMono(channelDataList, length) {
  const mono = new Float32Array(length);
  channelDataList.forEach((channel) => {
    for (let index = 0; index < length; index += 1) {
      mono[index] += channel[index] / channelDataList.length;
    }
  });
  return mono;
}

function extractBeatDataFromSamples(monoSamples, sampleRate, config) {
  if (!monoSamples.length || sampleRate <= 0) return [];
  const windowSize = Math.max(1, Math.floor(config.windowSize || 2048));
  const hopSize = Math.max(1, Math.floor(config.hopSize || 512));
  const alpha = Math.max(0, Math.min(1, Number(config.smoothingAlpha ?? 0.35)));
  if (monoSamples.length < windowSize) return [];
  const points = [];
  for (let start = 0; start + windowSize <= monoSamples.length; start += hopSize) {
    let sumSquares = 0;
    for (let index = start; index < start + windowSize; index += 1) {
      const sample = monoSamples[index];
      sumSquares += sample * sample;
    }
    points.push({ timeSeconds: start / sampleRate, strength: Math.sqrt(sumSquares / windowSize) });
  }
  let smoothed = points[0] ? points[0].strength : 0;
  for (let index = 1; index < points.length; index += 1) {
    smoothed = alpha * points[index].strength + (1 - alpha) * smoothed;
    points[index].strength = smoothed;
  }
  const maxStrength = points.reduce((max, point) => Math.max(max, point.strength), 0);
  return points.map((point) => ({ ...point, strength: maxStrength ? Math.max(0, Math.min(1, point.strength / maxStrength)) : 0 }));
}

function movingAverage(values, windowSize) {
  const safeWindow = Math.max(1, Math.floor(windowSize));
  const radius = Math.floor(safeWindow / 2);
  if (safeWindow <= 1 || values.length <= 2) return [...values];
  const smoothed = new Array(values.length).fill(0);
  for (let index = 0; index < values.length; index += 1) {
    const from = Math.max(0, index - radius);
    const to = Math.min(values.length - 1, index + radius);
    let sum = 0;
    let count = 0;
    for (let inner = from; inner <= to; inner += 1) {
      sum += values[inner];
      count += 1;
    }
    smoothed[index] = count ? sum / count : values[index];
  }
  return smoothed;
}

function approximateProminence(series, index, neighborhood) {
  const from = Math.max(0, index - neighborhood);
  const to = Math.min(series.length - 1, index + neighborhood);
  let leftMin = series[index];
  let rightMin = series[index];
  for (let pointer = from; pointer <= index; pointer += 1) leftMin = Math.min(leftMin, series[pointer]);
  for (let pointer = index; pointer <= to; pointer += 1) rightMin = Math.min(rightMin, series[pointer]);
  return Math.max(0, series[index] - Math.max(leftMin, rightMin));
}

function findZeroSlopePeakIndices(points, config = {}) {
  if (!points.length || points.length < 3) return [];
  const raw = points.map((point) => Number(point.strength || 0));
  const smooth = movingAverage(raw, 3);
  const minStrength = Math.max(0, Math.min(1, Number(config.minStrength ?? 0.16)));
  const minProminence = Math.max(0, Math.min(1, Number(config.minProminence ?? 0.05)));
  const minDistancePoints = Math.max(1, Math.floor(config.minDistancePoints || 3));
  const neighborhood = Math.max(2, minDistancePoints);
  const candidates = [];
  for (let index = 1; index < smooth.length - 1; index += 1) {
    const slopePrev = smooth[index] - smooth[index - 1];
    const slopeNext = smooth[index + 1] - smooth[index];
    if (!(slopePrev > 0 && slopeNext <= 0)) continue;
    if (smooth[index] < minStrength) continue;
    const prominence = approximateProminence(smooth, index, neighborhood);
    if (prominence < minProminence) continue;
    candidates.push({ index, strength: smooth[index] + prominence * 0.5 });
  }
  const byStrength = [...candidates].sort((left, right) => right.strength - left.strength);
  const accepted = [];
  byStrength.forEach((candidate) => {
    if (!accepted.some((row) => Math.abs(row.index - candidate.index) < minDistancePoints)) accepted.push(candidate);
  });
  return accepted.map((row) => row.index).sort((left, right) => left - right);
}

async function decodeRhythmAudio(path) {
  const cacheKey = String(path || "");
  if (state.rhythmAnalysisCache.has(cacheKey)) return state.rhythmAnalysisCache.get(cacheKey);
  const response = await fetch(`/api/audio?path=${encodeURIComponent(path)}`);
  if (!response.ok) throw new Error(`Could not load audio: ${response.status}`);
  const bytes = await response.arrayBuffer();
  const context = new (window.AudioContext || window.webkitAudioContext)();
  try {
    const buffer = await context.decodeAudioData(bytes.slice(0));
    state.rhythmAnalysisCache.set(cacheKey, buffer);
    return buffer;
  } finally {
    await context.close();
  }
}

function analysisSourceForProject(project, targetValue) {
  if (targetValue === "source") {
    return {
      sourceType: "source",
      sourceRef: "source",
      sourceLabel: (project.source || {}).label || "Source song",
      audioPath: (project.source || {}).audio_path || "",
    };
  }
  const trackId = String(targetValue || "").replace(/^track:/, "");
  const track = (project.tracks || []).find((item) => item.track_id === trackId);
  if (!track) throw new Error("Selected rhythm track was not found");
  return {
    sourceType: "track",
    sourceRef: track.track_id,
    sourceLabel: track.label || track.track_id,
    audioPath: track.audio_path || "",
  };
}

function sourceEventsFromPeaks(points, indices, sourceLabel) {
  return indices.map((index, order) => {
    const point = points[index];
    const nextPoint = points[indices[order + 1]] || null;
    const startSeconds = Number(point.timeSeconds || 0);
    const endSeconds = nextPoint ? Number(nextPoint.timeSeconds || startSeconds + 0.2) : startSeconds + 0.2;
    return {
      source: String(sourceLabel || "source")
        .toLowerCase()
        .replace(/[^a-z0-9_-]+/g, "_")
        .replace(/^_+|_+$/g, "") || "source",
      startSeconds,
      endSeconds,
      durationSeconds: Math.max(0.05, endSeconds - startSeconds),
      strength: Number(point.strength || 0),
    };
  });
}

async function runRhythmAnalysis() {
  const project = activeRhythmProject();
  if (!project) {
    showToast("Create or load a project first");
    return;
  }
  if (!project.source.audio_path) {
    showToast("Attach a source song before running analysis");
    return;
  }
  const target = analysisSourceForProject(project, el.rhythmAnalysisTarget.value || "source");
  state.activeRhythmPlaybackRef = target.sourceType === "track" ? `track:${target.sourceRef}` : "source";
  const config = rhythmAnalysisConfig();
  setPill(el.rhythmAnalysisState, "Analyzing", "warn");
  try {
    const buffer = await decodeRhythmAudio(target.audioPath);
    const channels = [];
    for (let channelIndex = 0; channelIndex < buffer.numberOfChannels; channelIndex += 1) {
      channels.push(buffer.getChannelData(channelIndex));
    }
    const mono = mixToMono(channels, buffer.length);
    const beatPoints = extractBeatDataFromSamples(mono, buffer.sampleRate, config);
    const minDistancePoints = Math.max(1, Math.round((config.minDistanceSeconds || 0.22) * buffer.sampleRate / Math.max(1, config.hopSize)));
    const peakIndices = findZeroSlopePeakIndices(beatPoints, {
      minStrength: config.minStrength,
      minProminence: config.minProminence,
      minDistancePoints,
    });
    const sourceEvents = sourceEventsFromPeaks(beatPoints, peakIndices, target.sourceLabel);
    const analysis = {
      analysis_id: `analysis-${Date.now().toString(16)}`,
      label: el.rhythmAnalysisLabel.value.trim() || target.sourceLabel,
      source_type: target.sourceType,
      source_ref: target.sourceRef,
      source_label: target.sourceLabel,
      algorithm: "energy_peaks_v1",
      settings: config,
      beat_points: peakIndices.map((index) => beatPoints[index]),
      source_events: sourceEvents,
      duration_seconds: buffer.duration,
      tempo_bpm: 0,
      created_at: new Date().toISOString(),
    };
    project.analyses = [analysis, ...(project.analyses || [])];
    state.selectedRhythmAnalysisId = analysis.analysis_id;
    await saveRhythmProject();
    setPill(el.rhythmAnalysisState, "Saved", "ok");
    showToast("Rhythm analysis saved");
  } catch (error) {
    setPill(el.rhythmAnalysisState, "Error", "error");
    showToast(error.message);
  }
}

function selectedRhythmRange() {
  const activeDraft = activeRhythmSelectionDraft();
  if (!activeDraft) return null;
  const ranges = (activeDraft.ranges || [])
    .map((range) => ({
      start: Math.min(range.startSeconds, range.endSeconds),
      end: Math.max(range.startSeconds, range.endSeconds),
    }))
    .filter((range) => range.end > range.start);
  if (!ranges.length) return null;
  return {
    start: Math.min(...ranges.map((range) => range.start)),
    end: Math.max(...ranges.map((range) => range.end)),
    ranges,
  };
}

async function saveRhythmSelection() {
  const project = activeRhythmProject();
  const analysis = currentRhythmAnalysis();
  const range = selectedRhythmRange();
  if (!project || !analysis) {
    showToast("Select an analysis first");
    return;
  }
  if (!range || range.end <= range.start) {
    showToast("Drag a range on the chart first");
    return;
  }
  const existingIndex = (project.selections || []).findIndex((selection) => selection.analysis_id === analysis.analysis_id);
  const existingSelection = existingIndex >= 0 ? project.selections[existingIndex] : null;
  const mergedRanges = normalizeRhythmSelectionRanges([
    ...((existingSelection?.ranges || []).map((segment) => ({
      start: Number(segment.start_seconds ?? segment.start ?? 0),
      end: Number(segment.end_seconds ?? segment.end ?? 0),
    }))),
    ...range.ranges.map((segment) => ({
      start: segment.start,
      end: segment.end,
    })),
  ]);
  if (!mergedRanges.length) {
    showToast("No valid selection ranges to save");
    return;
  }
  const beats = (analysis.beat_points || []).filter((beat) => {
    const time = Number(beat.timeSeconds || 0);
    return mergedRanges.some((segment) => time >= segment.start && time <= segment.end);
  });
  if (!beats.length) {
    showToast("No beats found inside that range");
    return;
  }
  const notes = (analysis.source_events || []).filter((event) => {
    const start = Number(event.startSeconds || 0);
    const end = Number(event.endSeconds || 0);
    return mergedRanges.some((segment) => end >= segment.start && start <= segment.end);
  });
  const labelInput = el.rhythmSelectionLabel.value.trim();
  const nextSelection = {
    selection_id: existingSelection ? existingSelection.selection_id : `selection-${Date.now().toString(16)}`,
    label: labelInput || existingSelection?.label || `${analysis.label} selection`,
    analysis_id: analysis.analysis_id,
    source: analysis.source_label,
    ranges: mergedRanges.map((segment) => ({
      start_seconds: segment.start,
      end_seconds: segment.end,
    })),
    range_start_seconds: mergedRanges[0].start,
    range_end_seconds: mergedRanges[mergedRanges.length - 1].end,
    game_beats: beats,
    game_notes: notes,
    game_beat_selections: mergedRanges.map((segment) => ({
      source: String(analysis.source_label || "source").toLowerCase().replace(/[^a-z0-9_-]+/g, "_"),
      startSeconds: segment.start,
      endSeconds: segment.end,
      minStrength: numericValue(el.rhythmMinStrength) ?? 0.16,
    })),
    game_beat_config: {
      gameMode: "step_arrows",
      mergeWindowSeconds: 0.12,
    },
    created_at: existingSelection?.created_at || new Date().toISOString(),
  };
  const nextSelections = [...(project.selections || [])];
  if (existingIndex >= 0) {
    nextSelections[existingIndex] = nextSelection;
  } else {
    nextSelections.unshift(nextSelection);
  }
  project.selections = nextSelections;
  delete state.rhythmSelectionDrafts[analysis.analysis_id];
  delete state.selectedRhythmDraftSegmentIndices[analysis.analysis_id];
  state.rhythmSelectionPointer = null;
  el.rhythmSelectionLabel.value = "";
  await saveRhythmProject();
  showToast("Beat selection saved");
}

function dedupeBeatPoints(points, mergeWindowSeconds = 0.12) {
  const sorted = [...points].sort((left, right) => Number(left.timeSeconds || 0) - Number(right.timeSeconds || 0));
  const merged = [];
  sorted.forEach((point) => {
    const time = Number(point.timeSeconds || 0);
    const strength = Number(point.strength || 0);
    const last = merged[merged.length - 1];
    if (last && Math.abs(Number(last.timeSeconds || 0) - time) <= mergeWindowSeconds) {
      if (strength > Number(last.strength || 0)) {
        last.timeSeconds = time;
        last.strength = strength;
      }
      return;
    }
    merged.push({ timeSeconds: time, strength });
  });
  return merged;
}

function normalizeRhythmSelectionRanges(ranges) {
  const sorted = (ranges || [])
    .map((range) => ({
      start: Math.min(Number(range.start ?? range.start_seconds ?? 0), Number(range.end ?? range.end_seconds ?? 0)),
      end: Math.max(Number(range.start ?? range.start_seconds ?? 0), Number(range.end ?? range.end_seconds ?? 0)),
    }))
    .filter((range) => range.end > range.start)
    .sort((left, right) => left.start - right.start);
  const merged = [];
  sorted.forEach((range) => {
    const last = merged[merged.length - 1];
    if (last && range.start <= last.end + 0.01) {
      last.end = Math.max(last.end, range.end);
      return;
    }
    merged.push({ start: range.start, end: range.end });
  });
  return merged;
}

async function mergeRhythmSelections() {
  const project = activeRhythmProject();
  if (!project) return;
  const selected = (project.selections || []).filter((selection) => state.selectedRhythmSelectionIds.includes(selection.selection_id));
  if (selected.length < 1) {
    showToast("Select at least one beat selection");
    return;
  }
  const mergeWindowSeconds = 0.12;
  const gameBeats = dedupeBeatPoints(selected.flatMap((selection) => selection.game_beats || []), mergeWindowSeconds);
  const gameNotes = selected.flatMap((selection) => selection.game_notes || []);
  const gameBeatSelections = selected.flatMap((selection) => selection.game_beat_selections || []);
  const merge = {
    merge_id: `merge-${Date.now().toString(16)}`,
    label: el.rhythmMergeLabel.value.trim() || `Merge ${new Date().toLocaleTimeString()}`,
    selection_ids: selected.map((selection) => selection.selection_id),
    game_beats: gameBeats,
    game_notes: gameNotes,
    game_beat_selections: gameBeatSelections,
    game_beat_config: {
      gameMode: "step_arrows",
      mergeWindowSeconds,
    },
    created_at: new Date().toISOString(),
  };
  project.merges = [merge, ...(project.merges || [])];
  state.selectedRhythmMergeId = merge.merge_id;
  el.rhythmMergeLabel.value = "";
  await saveRhythmProject();
  showToast("Merge candidate saved");
}

async function finalizeRhythmMerge() {
  const project = activeRhythmProject();
  const merge = currentRhythmMerge();
  if (!project || !merge) {
    showToast("Choose a merge candidate first");
    return;
  }
  project.final_result_id = merge.merge_id;
  await saveRhythmProject();
  showToast("Final rhythm beat asset saved to local library");
}

async function saveRhythmLyrics() {
  const project = activeRhythmProject();
  if (!project) return;
  project.lyrics = {
    ...(project.lyrics || {}),
    text: el.rhythmLyricsText.value,
    enabled: Boolean(el.rhythmLyricsText.value.trim()),
    source: "edited",
    updated_at_iso: new Date().toISOString(),
  };
  await saveRhythmProject(true);
}

async function runRhythmTrackExtraction() {
  const project = activeRhythmProject();
  if (!project) {
    showToast("Create or load a project first");
    return;
  }
  if (!project.source.audio_path) {
    showToast("Attach a source song before extracting tracks");
    return;
  }
  setPill(el.rhythmAnalysisState, "Extracting", "warn");
  applyAceRuntimeAvailability();
  try {
    const response = await api(`/api/rhythm-beats/projects/${encodeURIComponent(project.project_id)}/extract-track`, {
      method: "POST",
      body: JSON.stringify({
        track_name: el.rhythmExtractTrackName.value || "vocals",
        label: el.rhythmExtractTrackLabel.value.trim() || null,
        attach_to_project: true,
        guidance_scale: numericValue(el.rhythmExtractGuidanceScale) ?? 1.0,
      }),
    });
    if (response.project) {
      state.activeRhythmProject = response.project;
      state.activeRhythmProjectId = response.project.project_id;
    }
    await refreshExtractions();
    await refreshEditorAssets();
    await refreshRhythmProjects();
    const recoveryActive = Boolean(response.extraction && response.extraction.runtime_recovery && response.extraction.runtime_recovery.active);
    if (response.extraction && response.extraction.status === "recovering") {
      setPill(el.rhythmAnalysisState, "Recovering", "warn");
      startRuntimeRecoveryPolling();
      showToast(response.extraction.message);
    } else if (response.extraction && response.extraction.status === "complete") {
      el.rhythmExtractTrackLabel.value = "";
      if (recoveryActive) {
        setPill(el.rhythmAnalysisState, "Recovering", "warn");
        startRuntimeRecoveryPolling();
        showToast("Track extracted. ACE-Step is restarting to release memory.");
      } else {
        setPill(el.rhythmAnalysisState, "Extracted", "ok");
        showToast("Track extracted and linked to the project");
      }
    } else {
      setPill(el.rhythmAnalysisState, "Failed", "error");
      showToast((response.extraction && response.extraction.message) || "Track extraction failed");
    }
  } catch (error) {
    setPill(el.rhythmAnalysisState, "Error", "error");
    showToast(error.message);
  } finally {
    await refreshRuntimeState().catch(() => {});
    applyAceRuntimeAvailability();
  }
}

async function extractRhythmLyrics() {
  const project = activeRhythmProject();
  if (!project) {
    showToast("Create or load a project first");
    return;
  }
  if (!project.source.audio_path) {
    showToast("Attach a source song before extracting lyrics");
    return;
  }
  setPill(el.rhythmAssetState, "Extracting", "warn");
  try {
    const response = await api(`/api/rhythm-beats/projects/${encodeURIComponent(project.project_id)}/lyrics/extract`, {
      method: "POST",
      body: JSON.stringify({
        model: el.rhythmLyricsModel.value || "small",
        language: el.rhythmLyricsLanguage.value.trim() || null,
      }),
    });
    state.activeRhythmProject = response.project;
    state.activeRhythmProjectId = response.project.project_id;
    el.rhythmLyricsText.value = (response.lyrics && response.lyrics.text) || "";
    await refreshRhythmProjects();
    setPill(el.rhythmAssetState, "Lyrics ready", "ok");
    showToast("Lyrics extracted from source audio");
  } catch (error) {
    setPill(el.rhythmAssetState, "Error", "error");
    showToast(error.message);
  }
}

function renderLocalLibrary() {
  el.libraryList.replaceChildren();
  const items = filteredLibraryItems();
  setPill(el.libraryState, `${items.length} shown`, items.length ? "ok" : "neutral");
  el.libraryIndexPath.textContent = state.localLibraryIndexPath || "Index not created";
  el.librarySummary.innerHTML = [
    `<strong>${state.localLibraryItems.length} indexed items</strong>`,
    "The local library references existing Dance Station files in place.",
    "Use Reindex Creations after generating, editing, extracting, or training new assets.",
  ].join("<br>");
  renderLibraryConnection();

  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "empty-result";
    empty.textContent = state.localLibraryItems.length ? "No matching library items." : "No local library items yet. Reindex creations to build the local library.";
    el.libraryList.appendChild(empty);
    return;
  }

  items.forEach((item) => {
    const row = document.createElement("article");
    row.className = "library-item";
    row.dataset.itemId = item.id;
    const audioUrl = libraryAudioUrl(item);
    const primaryFile = itemAudioFile(item) || itemCoverFile(item) || {};
    const detailBadges = libraryDetailBadges(item);
    const publish = (item.metadata || {}).public_library || null;
    const isPublished = Boolean(publish && publish.remote_status === "published" && publish.remote_visibility === "public");
    const isPublishing = state.libraryPublishingItemIds.has(item.id);
    const isRevoking = state.libraryRevokingItemIds.has(item.id);
    const isRhythmGame = item.kind === "rhythm_game";
    const imported = Boolean((item.metadata || {}).imported);
    const creator = (item.metadata || {}).creator || {};
    const creatorName = creator.display_name || creator.creator_slug || "";
    const cardImage = libraryCardImageUrl(item);
    const rhythmMetadata = item.metadata || {};
    const rhythmVolumeId = String(rhythmMetadata.volume_id || "");
    const rhythmSupportedModes = rhythmMetadata.supported_game_modes || {};
    const rhythmStepArrowsEnabled = rhythmSupportedModes.step_arrows !== false;
    const rhythmOrbBeatEnabled = Boolean(rhythmSupportedModes.orb_beat);
    const canEditRhythmPublishSettings = isRhythmGame && !imported;
    row.innerHTML = `
      ${cardImage ? `<div class="library-card-art" style="background-image:url('${escapeHtml(cardImage)}')"></div>` : `<div class="library-card-art empty-art"></div>`}
      <div class="editor-asset-title">
        <strong>${escapeHtml(item.title)}</strong>
        <span class="category-badge">${escapeHtml(item.kind)}</span>
        ${imported ? `<span class="category-badge imported-badge">Imported</span>` : ""}
      </div>
      ${audioUrl ? `<audio controls preload="metadata" src="${audioUrl}"></audio>` : ""}
      <p class="asset-path">${escapeHtml(primaryFile.path || "No file path")}</p>
      <div class="library-meta-row">
        <span>${escapeHtml(item.status)}</span>
        <span>${escapeHtml(item.visibility)}</span>
        <span>${escapeHtml(formatLibraryDate(item.updated_at || item.created_at))}</span>
        ${creatorName ? `<span>Creator: ${escapeHtml(creatorName)}</span>` : ""}
        ${detailBadges.map((badge) => `<span>${escapeHtml(badge)}</span>`).join("")}
        ${isPublishing ? `<span>Publishing…</span>` : ""}
        ${isRevoking ? `<span>Revoking…</span>` : ""}
        ${publish ? `<span>Public: ${escapeHtml(publish.remote_status || "uploaded")}</span>` : ""}
      </div>
      <div class="control-grid library-edit-grid">
        <label class="field">
          <span>Title</span>
          <input class="library-title-input" type="text" value="${escapeHtml(item.title)}" />
        </label>
        <label class="field">
          <span>Tags</span>
          <input class="library-tags-input" type="text" value="${escapeHtml((item.tags || []).join(", "))}" placeholder="comma separated" />
        </label>
      </div>
      <label class="field">
        <span>Description</span>
        <textarea class="library-description-input" rows="2">${escapeHtml(item.description || "")}</textarea>
      </label>
      ${isRhythmGame ? `
        <div class="library-rhythm-publish-panel">
          <div class="generated-title">
            <strong>Game Publication</strong>
            <span>${canEditRhythmPublishSettings ? "Adjust before publish" : "Imported assets keep source settings"}</span>
          </div>
          <div class="control-grid">
            <label class="field">
              <span>Game availability</span>
              <select class="library-rhythm-enabled" aria-label="Game availability"${canEditRhythmPublishSettings ? "" : " disabled"}>
                <option value="true"${rhythmMetadata.game_enabled ? " selected" : ""}>Enabled in games</option>
                <option value="false"${!rhythmMetadata.game_enabled ? " selected" : ""}>Hidden from games</option>
              </select>
            </label>
            <label class="field">
              <span>Volume</span>
              <select class="library-rhythm-volume" aria-label="Rhythm-game volume"${canEditRhythmPublishSettings ? "" : " disabled"}>
                <option value="">No volume</option>
                ${assignableRhythmVolumeOptions(rhythmVolumeId).map((volume) => `<option value="${escapeHtml(volume.volume_id)}"${volume.volume_id === rhythmVolumeId ? " selected" : ""}>${escapeHtml(volume.label)}</option>`).join("")}
              </select>
            </label>
          </div>
          ${canEditRhythmPublishSettings ? `
            <div class="button-row generated-actions">
              <input class="library-rhythm-new-volume-input" type="text" placeholder="New volume name" />
              <button class="secondary-button library-rhythm-create-volume-button" type="button">Create Volume</button>
            </div>
          ` : ""}
          <div class="toggle-row rhythm-mode-toggle-row">
            <label><input class="library-rhythm-step-arrows" type="checkbox"${rhythmStepArrowsEnabled ? " checked" : ""}${canEditRhythmPublishSettings ? "" : " disabled"} /> Step arrows</label>
            <label><input class="library-rhythm-orb-beat" type="checkbox"${rhythmOrbBeatEnabled ? " checked" : ""}${canEditRhythmPublishSettings ? "" : " disabled"} /> Orb beat</label>
            <span class="mini-state">Laser shoot follows Step arrows</span>
          </div>
          ${canEditRhythmPublishSettings ? `
            <div class="button-row generated-actions">
              <button class="secondary-button library-rhythm-save-button" type="button">Save Game Settings</button>
            </div>
          ` : ""}
        </div>
      ` : ""}
      <div class="button-row generated-actions">
        <button class="secondary-button library-save-button" type="button">Save Metadata</button>
        <button class="secondary-button library-cover-button" type="button">Set Card Image</button>
        <button class="primary-button library-publish-button" type="button">${isPublishing ? (isPublished ? "Updating..." : "Publishing...") : (state.publicLibraryConnection?.authenticated ? (isPublished ? "Update Published" : "Publish") : "Connect wallet")}</button>
        ${isPublished ? `<button class="secondary-button library-revoke-button" type="button">Revoke</button>` : ""}
      </div>
    `;
    row.querySelector(".library-save-button").addEventListener("click", () => saveLibraryItem(row, item));
    row.querySelector(".library-cover-button").addEventListener("click", () => setLibraryCardImage(item));
    const rhythmCreateVolumeButton = row.querySelector(".library-rhythm-create-volume-button");
    if (rhythmCreateVolumeButton) {
      rhythmCreateVolumeButton.addEventListener("click", async () => {
        const input = row.querySelector(".library-rhythm-new-volume-input");
        const label = input ? input.value.trim() : "";
        if (!label) {
          showToast("Enter a new volume name");
          return;
        }
        try {
          rhythmCreateVolumeButton.disabled = true;
          rhythmCreateVolumeButton.textContent = "Creating...";
          const response = await api("/api/rhythm-beats/volumes", {
            method: "POST",
            body: JSON.stringify({ label }),
          });
          state.rhythmBeatVolumes = response.volumes || [];
          const created = (state.rhythmBeatVolumes || []).find((volume) => String(volume.label || "") === label);
          if (input) {
            input.value = "";
          }
          const select = row.querySelector(".library-rhythm-volume");
          if (select) {
            const currentValue = created ? created.volume_id : select.value;
            select.innerHTML = [
              `<option value="">No volume</option>`,
              ...assignableRhythmVolumeOptions(currentValue).map((volume) => `<option value="${escapeHtml(volume.volume_id)}"${volume.volume_id === currentValue ? " selected" : ""}>${escapeHtml(volume.label)}</option>`),
            ].join("");
          }
          showToast("Rhythm-game volume saved");
        } catch (error) {
          showToast(error.message);
        } finally {
          rhythmCreateVolumeButton.disabled = false;
          rhythmCreateVolumeButton.textContent = "Create Volume";
        }
      });
    }
    const rhythmSaveButton = row.querySelector(".library-rhythm-save-button");
    if (rhythmSaveButton) {
      rhythmSaveButton.addEventListener("click", async () => {
        try {
          rhythmSaveButton.disabled = true;
          rhythmSaveButton.textContent = "Saving...";
          await updateRhythmGameAssetSettings(item.id, {
            game_enabled: row.querySelector(".library-rhythm-enabled").value === "true",
            volume_id: row.querySelector(".library-rhythm-volume").value || null,
            step_arrows_enabled: row.querySelector(".library-rhythm-step-arrows").checked,
            orb_beat_enabled: row.querySelector(".library-rhythm-orb-beat").checked,
          });
        } catch (error) {
          showToast(error.message);
        } finally {
          rhythmSaveButton.disabled = false;
          rhythmSaveButton.textContent = "Save Game Settings";
        }
      });
    }
    const publishButton = row.querySelector(".library-publish-button");
    publishButton.disabled = !state.publicLibraryConnection?.authenticated || isPublishing || isRevoking;
    publishButton.addEventListener("click", () => publishLibraryItem(row, item));
    const revokeButton = row.querySelector(".library-revoke-button");
    if (revokeButton) {
      revokeButton.disabled = !state.publicLibraryConnection?.authenticated || isPublishing || isRevoking;
      if (isRevoking) revokeButton.textContent = "Revoking...";
      revokeButton.addEventListener("click", () => revokeLibraryItem(row, item));
    }
    el.libraryList.appendChild(row);
  });
}

function renderPublicLibrary() {
  el.publicLibraryList.replaceChildren();
  const items = filteredPublicLibraryItems();
  setPill(el.publicLibraryState, items.length ? `${items.length} public` : "Not loaded", items.length ? "ok" : "neutral");
  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "empty-result";
    empty.textContent = state.publicLibraryItems.length ? "No matching public assets." : "Load the public library to import items.";
    el.publicLibraryList.appendChild(empty);
    return;
  }
  items.forEach((item) => {
    const row = document.createElement("article");
    row.className = "library-item public-library-item";
    const creator = item.creator || {};
    const creatorName = creator.displayName || creator.creatorSlug || "Faceless creator";
    const cover = (item.files || []).find((file) => file.role === "cover");
    const audio = (item.files || []).find((file) => file.role === "audio" || file.role === "preview");
    const cardImage = (cover && cover.publicUrl) || creator.bannerUrl || creator.avatarUrl || "";
    row.innerHTML = `
      ${cardImage ? `<div class="library-card-art" style="background-image:url('${escapeHtml(cardImage)}')"></div>` : `<div class="library-card-art empty-art"></div>`}
      <div class="editor-asset-title">
        <strong>${escapeHtml(item.title)}</strong>
        <span class="category-badge">${escapeHtml(item.kind)}</span>
      </div>
      <div class="library-meta-row">
        <span>Creator: ${escapeHtml(creatorName)}</span>
        <span>${(item.files || []).length} files</span>
      </div>
      ${audio && audio.publicUrl ? `<audio controls preload="metadata" src="${escapeHtml(audio.publicUrl)}"></audio>` : ""}
      <div class="button-row generated-actions">
        <button class="primary-button public-import-button" type="button">Import</button>
      </div>
    `;
    row.querySelector(".public-import-button").addEventListener("click", () => importPublicLibraryItem(row, item));
    el.publicLibraryList.appendChild(row);
  });
}

function renderLibraryConnection() {
  const connection = state.publicLibraryConnection || {};
  if (el.libraryWalletProvider) {
    el.libraryWalletProvider.value = getPreferredLibraryWallet();
  }
  const selectedWallet = el.libraryWalletProvider ? el.libraryWalletProvider.value || "phantom" : "phantom";
  const walletAvailable = Boolean(getWalletProvider(selectedWallet));
  const authenticated = Boolean(connection.authenticated);
  setPill(
    el.libraryPublishState,
    authenticated ? "Connected" : "Disconnected",
    authenticated ? "ok" : "warn",
  );
  if (el.libraryWalletSummary) {
    if (authenticated) {
      el.libraryWalletSummary.textContent = `Connected as ${walletLabel(connection)}`;
    } else if (walletAvailable) {
      el.libraryWalletSummary.textContent = `Ready to connect with ${selectedWallet}.`;
    } else {
      el.libraryWalletSummary.textContent = `${selectedWallet} was not detected in this browser.`;
    }
  }
  el.connectLibraryWalletButton.disabled = false;
  el.disconnectLibraryWalletButton.disabled = !authenticated;
}

function formatLibraryDate(value) {
  if (!value) return "No date";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString();
}

function libraryDetailBadges(item) {
  const metadata = item.metadata || {};
  if (item.kind === "rhythm_game") {
    const supportedModes = metadata.supported_game_modes || {};
    const modeBadges = [];
    if (supportedModes.step_arrows !== false) modeBadges.push("Arrows");
    if (supportedModes.orb_beat) modeBadges.push("Orb");
    if (supportedModes.step_arrows !== false) modeBadges.push("Laser");
    return [
      metadata.game_enabled ? "Game enabled" : "Game hidden",
      metadata.volume_label ? `Volume: ${metadata.volume_label}` : "No volume",
      modeBadges.length ? `Modes: ${modeBadges.join(", ")}` : "Modes: none",
    ];
  }
  if (item.kind !== "dataset") return [];
  const declared = Number(metadata.sample_count || 0);
  const indexed = Number(metadata.indexed_sample_file_count || 0);
  const label = declared === indexed ? `${declared} samples` : `${declared} samples, ${indexed} files indexed`;
  return [label];
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
  const assets = filteredEditorAssets().filter((asset) => looksLikePlayableAudio(asset.audio_path));
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
  const selects = [el.sourceAssetSelect, el.extractSourceAssetSelect, el.instrumentAssetSelect, el.lokrAssetSelect].filter(Boolean);
  const audioAssets = state.editorAssets.filter((asset) => looksLikePlayableAudio(asset.audio_path));
  selects.forEach((select) => {
    const current = select.value;
    select.replaceChildren();
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Choose an existing creation";
    select.appendChild(placeholder);

    audioAssets.forEach((asset) => {
      const option = document.createElement("option");
      option.value = asset.asset_id;
      const imported = asset.imported ? " • imported" : "";
      const creator = asset.creator_name ? ` • ${asset.creator_name}` : "";
      option.textContent = `${asset.category}: ${asset.label}${imported}${creator}`;
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
  return voiceWorkAssetEntries().find((asset) => asset.asset_id === assetId) || null;
}

function activeDatasetEditorTarget() {
  return state.lokrDatasets.find((dataset) => dataset.metadata.dataset_id === state.datasetEditorTargetId) || null;
}

function datasetSourceSummary(source) {
  const metadata = source.metadata || {};
  return {
    label: metadata.label || metadata.name || source.label || source.dataset_id || source.library_item_id || "Dataset",
    count: Number(source.sample_count || (source.samples || []).length || 0),
  };
}

function cloneDatasetSampleForTarget(sample, donor) {
  const cloned = typeof structuredClone === "function" ? structuredClone(sample) : JSON.parse(JSON.stringify(sample));
  const id = window.crypto && window.crypto.randomUUID ? window.crypto.randomUUID() : `sample-${Date.now()}-${Math.random().toString(16).slice(2, 10)}`;
  const donorMeta = donor.metadata || {};
  cloned.id = id;
  cloned.source_dataset_id = donor.dataset_id || donor.library_item_id || donorMeta.dataset_id || "";
  cloned.source_dataset_source_id = donor.source_id || "";
  cloned.source_entry_id = sample.id || "";
  cloned.labeled = Boolean(cloned.labeled || cloned.caption || cloned.label);
  return cloned;
}

function datasetMissingAudioCount(dataset) {
  return Number((dataset && dataset.validation && dataset.validation.missing_audio_count) || 0);
}

function activeLokrDataset() {
  return state.lokrDatasets.find((dataset) => dataset.metadata.dataset_id === state.activeLokrDatasetId) || null;
}

function setActiveLokrDataset(dataset) {
  if (!dataset) return;
  const existingIndex = state.lokrDatasets.findIndex((item) => item.metadata.dataset_id === dataset.metadata.dataset_id);
  if (existingIndex >= 0) {
    state.lokrDatasets[existingIndex] = dataset;
  } else {
    state.lokrDatasets.unshift(dataset);
  }
  state.activeLokrDatasetId = dataset.metadata.dataset_id;
  renderLokrDatasets();
  renderLokrDatasetEditor();
  renderLokrRuns();
}

function renderLokrDatasets() {
  el.lokrDatasetList.replaceChildren();
  setPill(el.lokrDatasetState, `${state.lokrDatasets.length} datasets`, state.lokrDatasets.length ? "ok" : "neutral");
  if (!state.lokrDatasets.length) {
    const empty = document.createElement("div");
    empty.className = "empty-result";
    empty.textContent = "No LoKr datasets yet.";
    el.lokrDatasetList.appendChild(empty);
    return;
  }
  state.lokrDatasets.forEach((dataset) => {
    const metadata = dataset.metadata || {};
    const row = document.createElement("article");
    row.className = `generated-item lokr-dataset-item${metadata.dataset_id === state.activeLokrDatasetId ? " active" : ""}`;
    row.innerHTML = `
      <div class="generated-title">
        <strong>${escapeHtml(metadata.label || metadata.name || "LoKr dataset")}</strong>
        <span>${Number(metadata.num_samples || 0)} samples</span>
      </div>
      <div class="asset-path">${escapeHtml(metadata.custom_tag ? `Trigger: ${metadata.custom_tag}` : "No trigger tag")}</div>
      <button class="secondary-button full-width" type="button">Open Dataset</button>
    `;
    row.querySelector("button").addEventListener("click", async () => {
      const response = await api(`/api/lokr/datasets/${encodeURIComponent(metadata.dataset_id)}`);
      setActiveLokrDataset(response);
    });
    el.lokrDatasetList.appendChild(row);
  });
}

function renderLokrDatasetEditor() {
  const dataset = activeLokrDataset();
  el.lokrEntryList.replaceChildren();
  if (!dataset) {
    el.lokrActiveDatasetReadout.textContent = "No dataset selected";
    el.lokrDatasetLabel.value = "";
    el.lokrCustomTag.value = "";
    el.lokrDefaultGenre.value = "";
    el.lokrDefaultLanguage.value = "unknown";
    el.lokrTagPosition.value = "prepend";
    el.lokrGenreRatio.value = "0";
    el.lokrSampleCount.value = "";
    el.lokrAllInstrumental.checked = true;
    el.saveLokrDatasetButton.disabled = true;
    setPill(el.lokrValidationState, "No dataset", "neutral");
    el.lokrDatasetSummary.textContent = "Create or select a dataset to begin adding songs.";
    const empty = document.createElement("div");
    empty.className = "empty-result";
    empty.textContent = "Create or select a dataset to add songs.";
    el.lokrEntryList.appendChild(empty);
    return;
  }

  const metadata = dataset.metadata || {};
  const samples = dataset.samples || [];
  const missingAudio = datasetMissingAudioCount(dataset);
  el.lokrActiveDatasetReadout.textContent = metadata.dataset_id;
  el.lokrDatasetLabel.value = metadata.label || metadata.name || "";
  el.lokrCustomTag.value = metadata.custom_tag || "";
  el.lokrDefaultGenre.value = metadata.default_genre || "";
  el.lokrDefaultLanguage.value = metadata.default_language || "unknown";
  el.lokrTagPosition.value = metadata.tag_position || "prepend";
  el.lokrGenreRatio.value = String(metadata.genre_ratio ?? 0);
  el.lokrSampleCount.value = `${samples.length}`;
  el.lokrAllInstrumental.checked = Boolean(metadata.all_instrumental);
  el.saveLokrDatasetButton.disabled = false;

  const missingCaptions = samples.filter((sample) => !(sample.caption || "").trim()).length;
  const tone = samples.length && !missingCaptions && !missingAudio ? "ok" : samples.length ? "warn" : "neutral";
  setPill(el.lokrValidationState, samples.length ? `${samples.length} samples` : "Empty", tone);
  el.lokrDatasetSummary.innerHTML = [
    `<strong>${escapeHtml(metadata.label || "LoKr dataset")}</strong>`,
    `Samples: ${samples.length}`,
    `Missing audio: ${missingAudio}`,
    `Missing captions: ${missingCaptions}`,
    `JSON: ${escapeHtml(dataset.metadata_path || "")}`,
  ].join("<br>");

  if (!samples.length) {
    const empty = document.createElement("div");
    empty.className = "empty-result";
    empty.textContent = "No songs in this dataset yet.";
    el.lokrEntryList.appendChild(empty);
    return;
  }

  samples.forEach((sample, index) => {
    const item = document.createElement("article");
    item.className = "lokr-entry generated-item";
    item.dataset.entryId = sample.id;
    item.innerHTML = `
      <div class="generated-title">
        <strong>${escapeHtml(sample.label || sample.filename || `Sample ${index + 1}`)}</strong>
        <span>${sample.has_audio ? (sample.duration ? `${Number(sample.duration).toFixed(1)}s` : "duration unknown") : "audio missing"}</span>
      </div>
      ${sample.audio_url ? `<audio controls preload="metadata" src="${sample.audio_url}"></audio>` : ""}
      <div class="control-grid">
        <label class="field">
          <span>Label</span>
          <input class="lokr-entry-label" type="text" value="${escapeHtml(sample.label || "")}" />
        </label>
        <label class="field">
          <span>Genre</span>
          <input class="lokr-entry-genre" type="text" value="${escapeHtml(sample.genre || "")}" placeholder="optional" />
        </label>
        <label class="field">
          <span>Language</span>
          <input class="lokr-entry-language" type="text" value="${escapeHtml(sample.language || "unknown")}" />
        </label>
        <label class="field">
          <span>BPM</span>
          <input class="lokr-entry-bpm" type="text" value="${escapeHtml(String(sample.bpm ?? "N/A"))}" />
        </label>
        <label class="field">
          <span>Key</span>
          <input class="lokr-entry-keyscale" type="text" value="${escapeHtml(sample.keyscale || "N/A")}" />
        </label>
        <label class="field">
          <span>Time signature</span>
          <input class="lokr-entry-timesignature" type="text" value="${escapeHtml(sample.timesignature || "4")}" />
        </label>
      </div>
      <label class="field">
        <span>Caption</span>
        <textarea class="lokr-entry-caption" rows="3" placeholder="Detailed description of the audible style and instrumentation">${escapeHtml(sample.caption || "")}</textarea>
      </label>
      <label class="field">
        <span>Lyrics</span>
        <textarea class="lokr-entry-lyrics" rows="4">${escapeHtml(sample.lyrics || "[Instrumental]")}</textarea>
      </label>
      <div class="control-grid">
        <label class="field">
          <span>Trigger tag override</span>
          <input class="lokr-entry-custom-tag" type="text" value="${escapeHtml(sample.custom_tag || "")}" />
        </label>
        <label class="field">
          <span>Prompt override</span>
          <select class="lokr-entry-prompt-override">
            <option value="">Dataset default</option>
            <option value="caption"${sample.prompt_override === "caption" ? " selected" : ""}>Caption</option>
            <option value="genre"${sample.prompt_override === "genre" ? " selected" : ""}>Genre</option>
          </select>
        </label>
      </div>
      <div class="toggle-row">
        <label><input class="lokr-entry-instrumental" type="checkbox"${sample.is_instrumental ? " checked" : ""} /> Instrumental</label>
        <label><input class="lokr-entry-labeled" type="checkbox"${sample.labeled ? " checked" : ""} /> Labeled</label>
      </div>
      <div class="button-row generated-actions">
        <button class="secondary-button lokr-attach-file-button" type="button">Attach File</button>
        <button class="secondary-button lokr-attach-asset-button" type="button">Use Selected Creation</button>
        <button class="secondary-button lokr-delete-entry-button" type="button">Delete Entry</button>
      </div>
    `;
    item.querySelector(".lokr-attach-file-button").addEventListener("click", () => attachFileToLokrEntry(sample.id));
    item.querySelector(".lokr-attach-asset-button").addEventListener("click", () => attachAssetToLokrEntry(sample.id));
    item.querySelector(".lokr-delete-entry-button").addEventListener("click", () => deleteLokrEntry(sample.id));
    el.lokrEntryList.appendChild(item);
  });
}

function lokrDatasetFromEditor() {
  const dataset = activeLokrDataset();
  if (!dataset) return null;
  const metadata = {
    ...(dataset.metadata || {}),
    label: el.lokrDatasetLabel.value.trim() || "LoKr dataset",
    name: el.lokrDatasetLabel.value.trim() || "LoKr dataset",
    custom_tag: el.lokrCustomTag.value.trim(),
    default_genre: el.lokrDefaultGenre.value.trim(),
    default_language: el.lokrDefaultLanguage.value.trim() || "unknown",
    tag_position: el.lokrTagPosition.value,
    genre_ratio: Number(el.lokrGenreRatio.value || 0),
    all_instrumental: el.lokrAllInstrumental.checked,
  };
  const samples = [...el.lokrEntryList.querySelectorAll(".lokr-entry")].map((row) => {
    const original = (dataset.samples || []).find((sample) => sample.id === row.dataset.entryId) || {};
    const instrumental = row.querySelector(".lokr-entry-instrumental").checked;
    const lyrics = row.querySelector(".lokr-entry-lyrics").value.trim();
    const genre = row.querySelector(".lokr-entry-genre").value.trim() || metadata.default_genre || "";
    const language = row.querySelector(".lokr-entry-language").value.trim() || metadata.default_language || "unknown";
    return {
      ...original,
      label: row.querySelector(".lokr-entry-label").value.trim(),
      caption: row.querySelector(".lokr-entry-caption").value.trim(),
      genre,
      lyrics: instrumental ? "[Instrumental]" : lyrics || "[Instrumental]",
      formatted_lyrics: instrumental ? "[Instrumental]" : lyrics || "[Instrumental]",
      bpm: row.querySelector(".lokr-entry-bpm").value.trim() || "N/A",
      keyscale: row.querySelector(".lokr-entry-keyscale").value.trim() || "N/A",
      timesignature: row.querySelector(".lokr-entry-timesignature").value.trim() || "4",
      language,
      custom_tag: row.querySelector(".lokr-entry-custom-tag").value.trim(),
      prompt_override: row.querySelector(".lokr-entry-prompt-override").value || null,
      is_instrumental: instrumental,
      labeled: row.querySelector(".lokr-entry-labeled").checked,
    };
  });
  return { metadata, samples };
}

function datasetEditorTargetFromEditor() {
  const dataset = activeDatasetEditorTarget();
  if (!dataset) return null;
  const metadata = {
    ...(dataset.metadata || {}),
    label: el.datasetEditorLabel.value.trim() || "LoKr dataset",
    name: el.datasetEditorLabel.value.trim() || "LoKr dataset",
    custom_tag: el.datasetEditorCustomTag.value.trim(),
    default_genre: el.datasetEditorDefaultGenre.value.trim(),
    default_language: el.datasetEditorDefaultLanguage.value.trim() || "unknown",
    tag_position: el.datasetEditorTagPosition.value,
    genre_ratio: Number(el.datasetEditorGenreRatio.value || 0),
    all_instrumental: el.datasetEditorAllInstrumental.checked,
  };
  const samples = [...el.datasetEditorEntryList.querySelectorAll(".lokr-entry")].map((row) => {
    const original = (dataset.samples || []).find((sample) => sample.id === row.dataset.entryId) || {};
    const instrumental = row.querySelector(".dataset-editor-entry-instrumental").checked;
    const lyrics = row.querySelector(".dataset-editor-entry-lyrics").value.trim();
    const genre = row.querySelector(".dataset-editor-entry-genre").value.trim() || metadata.default_genre || "";
    const language = row.querySelector(".dataset-editor-entry-language").value.trim() || metadata.default_language || "unknown";
    return {
      ...original,
      label: row.querySelector(".dataset-editor-entry-label").value.trim(),
      caption: row.querySelector(".dataset-editor-entry-caption").value.trim(),
      genre,
      lyrics: instrumental ? "[Instrumental]" : lyrics || "[Instrumental]",
      formatted_lyrics: instrumental ? "[Instrumental]" : lyrics || "[Instrumental]",
      bpm: row.querySelector(".dataset-editor-entry-bpm").value.trim() || "N/A",
      keyscale: row.querySelector(".dataset-editor-entry-keyscale").value.trim() || "N/A",
      timesignature: row.querySelector(".dataset-editor-entry-timesignature").value.trim() || "4",
      language,
      custom_tag: row.querySelector(".dataset-editor-entry-custom-tag").value.trim(),
      prompt_override: row.querySelector(".dataset-editor-entry-prompt-override").value || null,
      is_instrumental: instrumental,
      labeled: row.querySelector(".dataset-editor-entry-labeled").checked,
    };
  });
  return { metadata, samples };
}

async function createLokrDataset() {
  const label = el.lokrNewDatasetLabel.value.trim() || "New LoKr dataset";
  const response = await api("/api/lokr/datasets", {
    method: "POST",
    body: JSON.stringify({ label, default_genre: "", default_language: "unknown" }),
  });
  setActiveLokrDataset(response.dataset);
  showToast("LoKr dataset created");
}

function selectedImportJsonFile(input) {
  return input && input.files && input.files[0] ? input.files[0] : null;
}

async function createLokrDatasetFromJson() {
  const file = selectedImportJsonFile(el.lokrDatasetJsonFile);
  if (!file) {
    showToast("Choose a dataset JSON file");
    return;
  }
  const formData = new FormData();
  formData.append("file", file, file.name);
  formData.append("label", el.lokrNewDatasetLabel.value.trim() || "Imported LoKr dataset");
  const response = await fetch("/api/lokr/datasets/import-json", { method: "POST", body: formData });
  const body = await response.json().catch(() => null);
  if (!response.ok) throw new Error(formatApiDetail(body && body.detail ? body.detail : `Import failed: ${response.status}`));
  setActiveLokrDataset(body.dataset);
  await refreshDatasetSources();
  showToast("Dataset JSON imported");
}

async function appendLokrDatasetJson() {
  const dataset = activeLokrDataset();
  const file = selectedImportJsonFile(el.lokrDatasetJsonFile);
  if (!dataset) {
    showToast("Select a LoKr dataset first");
    return;
  }
  if (!file) {
    showToast("Choose a dataset JSON file");
    return;
  }
  const formData = new FormData();
  formData.append("file", file, file.name);
  const response = await fetch(`/api/lokr/datasets/${encodeURIComponent(dataset.metadata.dataset_id)}/import-json`, {
    method: "POST",
    body: formData,
  });
  const body = await response.json().catch(() => null);
  if (!response.ok) throw new Error(formatApiDetail(body && body.detail ? body.detail : `Import failed: ${response.status}`));
  setActiveLokrDataset(body.dataset);
  await refreshDatasetSources();
  showToast("JSON entries appended to dataset");
}

async function addEmptyLokrEntry() {
  const dataset = activeLokrDataset();
  if (!dataset) {
    showToast("Create or select a LoKr dataset first");
    return;
  }
  const response = await api(`/api/lokr/datasets/${encodeURIComponent(dataset.metadata.dataset_id)}/entries/empty`, {
    method: "POST",
  });
  setActiveLokrDataset(response.dataset);
  showToast("Empty dataset entry added");
}

async function createDatasetEditorTarget() {
  const label = el.datasetEditorNewLabel.value.trim() || "New LoKr dataset";
  const response = await api("/api/lokr/datasets", {
    method: "POST",
    body: JSON.stringify({ label, default_genre: "", default_language: "unknown" }),
  });
  setActiveLokrDataset(response.dataset);
  state.datasetEditorTargetId = response.dataset.metadata.dataset_id;
  await refreshDatasetSources();
  renderDatasetEditorTarget();
  showToast("Target dataset created");
}

async function createDatasetEditorTargetFromJson() {
  const file = selectedImportJsonFile(el.datasetEditorJsonFile);
  if (!file) {
    showToast("Choose a dataset JSON file");
    return;
  }
  const formData = new FormData();
  formData.append("file", file, file.name);
  formData.append("label", el.datasetEditorNewLabel.value.trim() || "Imported LoKr dataset");
  const response = await fetch("/api/lokr/datasets/import-json", { method: "POST", body: formData });
  const body = await response.json().catch(() => null);
  if (!response.ok) throw new Error(formatApiDetail(body && body.detail ? body.detail : `Import failed: ${response.status}`));
  setActiveLokrDataset(body.dataset);
  state.datasetEditorTargetId = body.dataset.metadata.dataset_id;
  await refreshDatasetSources();
  renderDatasetEditorTarget();
  showToast("Target dataset imported from JSON");
}

async function appendDatasetEditorJson() {
  const dataset = activeDatasetEditorTarget();
  const file = selectedImportJsonFile(el.datasetEditorJsonFile);
  if (!dataset) {
    showToast("Select a target dataset first");
    return;
  }
  if (!file) {
    showToast("Choose a dataset JSON file");
    return;
  }
  const formData = new FormData();
  formData.append("file", file, file.name);
  const response = await fetch(`/api/lokr/datasets/${encodeURIComponent(dataset.metadata.dataset_id)}/import-json`, {
    method: "POST",
    body: formData,
  });
  const body = await response.json().catch(() => null);
  if (!response.ok) throw new Error(formatApiDetail(body && body.detail ? body.detail : `Import failed: ${response.status}`));
  setActiveLokrDataset(body.dataset);
  state.datasetEditorTargetId = body.dataset.metadata.dataset_id;
  await refreshDatasetSources();
  renderDatasetEditorTarget();
  showToast("JSON entries appended to target dataset");
}

async function refreshLokrDatasets() {
  state.lokrDatasets = await api("/api/lokr/datasets");
  if (state.activeLokrDatasetId && !state.lokrDatasets.some((dataset) => dataset.metadata.dataset_id === state.activeLokrDatasetId)) {
    state.activeLokrDatasetId = null;
  }
  renderLokrDatasets();
  renderLokrDatasetEditor();
}

async function refreshDatasetSources() {
  const [datasets, sources] = await Promise.all([api("/api/lokr/datasets"), api("/api/lokr/dataset-sources")]);
  state.lokrDatasets = datasets;
  state.datasetSources = sources;
  if (state.activeLokrDatasetId && !state.lokrDatasets.some((dataset) => dataset.metadata.dataset_id === state.activeLokrDatasetId)) {
    state.activeLokrDatasetId = null;
  }
  if (state.datasetEditorTargetId && !state.lokrDatasets.some((dataset) => dataset.metadata.dataset_id === state.datasetEditorTargetId)) {
    state.datasetEditorTargetId = null;
  }
  if (state.datasetEditorDonorSourceId && !state.datasetSources.some((source) => source.source_id === state.datasetEditorDonorSourceId)) {
    state.datasetEditorDonorSourceId = null;
    state.datasetEditorDonor = null;
  }
  renderLokrDatasets();
  renderLokrDatasetEditor();
  renderDatasetEditorSources();
  renderDatasetEditorTarget();
  renderDatasetEditorDonor();
}

async function saveLokrDataset() {
  const dataset = lokrDatasetFromEditor();
  if (!dataset) {
    showToast("Select a LoKr dataset");
    return;
  }
  const datasetId = activeLokrDataset().metadata.dataset_id;
  const response = await api(`/api/lokr/datasets/${encodeURIComponent(datasetId)}`, {
    method: "POST",
    body: JSON.stringify({ dataset }),
  });
  setActiveLokrDataset(response.dataset);
  const missingAudio = datasetMissingAudioCount(response.dataset);
  showToast(missingAudio ? `LoKr dataset saved with ${missingAudio} entr${missingAudio === 1 ? "y" : "ies"} missing audio` : "LoKr dataset saved");
}

async function saveDatasetEditorTarget({ silent = false } = {}) {
  const dataset = datasetEditorTargetFromEditor();
  const active = activeDatasetEditorTarget();
  if (!dataset || !active) {
    showToast("Select a target dataset");
    return null;
  }
  const response = await api(`/api/lokr/datasets/${encodeURIComponent(active.metadata.dataset_id)}`, {
    method: "POST",
    body: JSON.stringify({ dataset }),
  });
  setActiveLokrDataset(response.dataset);
  state.datasetEditorTargetId = response.dataset.metadata.dataset_id;
  await refreshDatasetSources();
  renderDatasetEditorTarget();
  if (!silent) {
    const missingAudio = datasetMissingAudioCount(response.dataset);
    showToast(missingAudio ? `Target dataset saved with ${missingAudio} entr${missingAudio === 1 ? "y" : "ies"} missing audio` : "Target dataset saved");
  }
  return response.dataset;
}

async function loadDatasetEditorDonor(sourceId) {
  const donor = await api(`/api/lokr/dataset-sources/${encodeURIComponent(sourceId)}`);
  state.datasetEditorDonorSourceId = donor.source_id;
  state.datasetEditorDonor = donor;
  renderDatasetEditorSources();
  renderDatasetEditorDonor();
}

async function pushDonorEntryIntoTarget(sample) {
  const target = activeDatasetEditorTarget();
  const donor = state.datasetEditorDonor;
  if (!target || !donor) {
    showToast("Choose both a target dataset and a donor dataset");
    return;
  }
  if (donor.source_kind === "local" && donor.dataset_id === target.metadata.dataset_id) {
    showToast("Choose a different donor dataset");
    return;
  }
  target.samples = [...(target.samples || []), cloneDatasetSampleForTarget(sample, donor)];
  renderDatasetEditorTarget();
  await saveDatasetEditorTarget({ silent: true });
  showToast("Entry pushed into target dataset");
}

async function deleteDatasetEditorTargetEntry(entryId) {
  const target = activeDatasetEditorTarget();
  if (!target) return;
  target.samples = (target.samples || []).filter((sample) => sample.id !== entryId);
  renderDatasetEditorTarget();
  await saveDatasetEditorTarget({ silent: true });
  showToast("Target dataset entry deleted");
}

async function uploadLokrFiles(files) {
  const dataset = activeLokrDataset();
  if (!dataset) {
    showToast("Create or select a LoKr dataset first");
    return;
  }
  const fileList = [...files].filter(Boolean);
  if (!fileList.length) return;
  setPill(el.lokrEntryState, "Adding", "warn");
  let latest = dataset;
  try {
    for (const file of fileList) {
      const formData = new FormData();
      formData.append("file", file);
      const response = await fetch(`/api/lokr/datasets/${encodeURIComponent(dataset.metadata.dataset_id)}/entries/upload`, {
        method: "POST",
        body: formData,
      });
      const body = await response.json().catch(() => null);
      if (!response.ok) throw new Error(formatApiDetail(body && body.detail ? body.detail : `Upload failed: ${response.status}`));
      latest = body.dataset;
    }
    setActiveLokrDataset(latest);
    setPill(el.lokrEntryState, "Added", "ok");
    showToast("Audio added to LoKr dataset");
  } catch (error) {
    setPill(el.lokrEntryState, "Error", "error");
    showToast(error.message);
  }
}

async function addLokrAsset() {
  const dataset = activeLokrDataset();
  const asset = selectedSourceAsset(el.lokrAssetSelect);
  if (!dataset || !asset) {
    showToast("Choose a dataset and creation");
    return;
  }
  const response = await api(`/api/lokr/datasets/${encodeURIComponent(dataset.metadata.dataset_id)}/entries/from-asset`, {
    method: "POST",
    body: JSON.stringify({ asset_id: asset.asset_id }),
  });
  setActiveLokrDataset(response.dataset);
  showToast("Creation added to LoKr dataset");
}

async function attachFileToLokrEntry(entryId) {
  const dataset = activeLokrDataset();
  if (!dataset) return;
  const input = document.createElement("input");
  input.type = "file";
  input.accept = ".mp3,.wav,.flac,.ogg,.m4a,.opus,audio/*";
  input.addEventListener("change", async () => {
    const file = input.files && input.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    try {
      const response = await fetch(`/api/lokr/datasets/${encodeURIComponent(dataset.metadata.dataset_id)}/entries/${encodeURIComponent(entryId)}/upload`, {
        method: "POST",
        body: formData,
      });
      const body = await response.json().catch(() => null);
      if (!response.ok) throw new Error(formatApiDetail(body && body.detail ? body.detail : `Upload failed: ${response.status}`));
      setActiveLokrDataset(body.dataset);
      showToast("Audio attached to dataset entry");
    } catch (error) {
      showToast(error.message);
    }
  });
  input.click();
}

async function attachAssetToLokrEntry(entryId) {
  const dataset = activeLokrDataset();
  const asset = selectedSourceAsset(el.lokrAssetSelect);
  if (!dataset || !asset) {
    showToast("Choose an existing creation first");
    return;
  }
  const response = await api(`/api/lokr/datasets/${encodeURIComponent(dataset.metadata.dataset_id)}/entries/attach-asset`, {
    method: "POST",
    body: JSON.stringify({ entry_id: entryId, asset_id: asset.asset_id }),
  });
  setActiveLokrDataset(response.dataset);
  showToast("Creation attached to dataset entry");
}

async function attachFileToDatasetEditorEntry(entryId) {
  const dataset = activeDatasetEditorTarget();
  if (!dataset) return;
  const input = document.createElement("input");
  input.type = "file";
  input.accept = ".mp3,.wav,.flac,.ogg,.m4a,.opus,audio/*";
  input.addEventListener("change", async () => {
    const file = input.files && input.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    try {
      const response = await fetch(`/api/lokr/datasets/${encodeURIComponent(dataset.metadata.dataset_id)}/entries/${encodeURIComponent(entryId)}/upload`, {
        method: "POST",
        body: formData,
      });
      const body = await response.json().catch(() => null);
      if (!response.ok) throw new Error(formatApiDetail(body && body.detail ? body.detail : `Upload failed: ${response.status}`));
      setActiveLokrDataset(body.dataset);
      state.datasetEditorTargetId = body.dataset.metadata.dataset_id;
      await refreshDatasetSources();
      renderDatasetEditorTarget();
      showToast("Audio attached to target dataset entry");
    } catch (error) {
      showToast(error.message);
    }
  });
  input.click();
}

async function deleteLokrEntry(entryId) {
  const dataset = activeLokrDataset();
  if (!dataset) return;
  const response = await api(
    `/api/lokr/datasets/${encodeURIComponent(dataset.metadata.dataset_id)}/entries/${encodeURIComponent(entryId)}`,
    { method: "DELETE" },
  );
  setActiveLokrDataset(response.dataset);
  showToast("LoKr dataset entry deleted");
}

function lokrRunPayload() {
  return {
    model: el.lokrTrainModel.value,
    sidestep_command: el.lokrSidestepCommand.value.trim() || "uv run sidestep",
    checkpoint_dir: el.lokrCheckpointDir.value.trim() || "runtimes/ACE-Step-1.5/checkpoints",
  };
}

function lokrTrainingPayload() {
  return {
    ...lokrRunPayload(),
    epochs: numericValue(el.lokrTrainEpochs),
    lokr_linear_dim: numericValue(el.lokrTrainDim),
    lokr_linear_alpha: numericValue(el.lokrTrainAlpha),
    save_every: numericValue(el.lokrTrainSaveEvery),
    optimizer_type: el.lokrTrainOptimizer.value,
    batch_size: numericValue(el.lokrTrainBatchSize),
    gradient_accumulation: numericValue(el.lokrTrainGradAccum),
    gradient_checkpointing: el.lokrGradientCheckpointing.checked,
    offload_encoder: el.lokrOffloadEncoder.checked,
    chunk_duration: numericValue(el.lokrTrainChunkDuration),
  };
}

function latestLokrPreprocessRun(datasetId) {
  return state.lokrRuns.find((run) => run.dataset_id === datasetId && run.type === "preprocess" && run.status === "complete" && run.ready_to_train);
}

function isLokrRunVisible(run) {
  if (!state.lokrRunViewClearedAt) return true;
  if (run.status === "running") return true;
  const createdAt = Date.parse(run.created_at || "");
  if (!Number.isFinite(createdAt)) return true;
  return createdAt >= state.lokrRunViewClearedAt;
}

function visibleLokrRuns() {
  return state.lokrRuns.filter(isLokrRunVisible);
}

function activeLokrRun() {
  return state.lokrRuns.find((run) => run.status === "running") || null;
}

function lokrProgressDetails(run) {
  const details = [];
  if (run.current_epoch !== undefined && run.current_epoch !== null) {
    details.push(`Epoch ${run.current_epoch}${run.max_epochs ? `/${run.max_epochs}` : ""}`);
  }
  if (run.current_step !== undefined && run.current_step !== null) {
    details.push(`Step ${run.current_step}`);
  }
  if (run.loss !== undefined && run.loss !== null) {
    const loss = Number(run.loss);
    details.push(Number.isFinite(loss) ? `Loss ${loss.toFixed(4)}` : `Loss ${run.loss}`);
  }
  return details;
}

function renderLokrTrainingReadiness() {
  const dataset = activeLokrDataset();
  const activeRun = activeLokrRun();
  el.preprocessLokrButton.disabled = Boolean(activeRun) || !dataset;
  el.stopLokrRunButton.disabled = !activeRun;
  if (!dataset) {
    el.lokrTrainingReadiness.textContent = "Select a dataset and preprocess it before training.";
    el.trainLokrButton.disabled = true;
    return;
  }
  if (activeRun) {
    const details = lokrProgressDetails(activeRun);
    const progress = details.length ? ` ${details.join(" | ")}.` : activeRun.summary ? ` ${activeRun.summary}.` : "";
    el.lokrTrainingReadiness.textContent = `${activeRun.label || activeRun.run_id} is running.${progress}`;
    el.trainLokrButton.disabled = true;
    return;
  }
  const datasetId = dataset.metadata.dataset_id;
  const runningPreprocess = state.lokrRuns.find((run) => run.dataset_id === datasetId && run.type === "preprocess" && run.status === "running");
  if (runningPreprocess) {
    const summary = runningPreprocess.summary ? ` ${runningPreprocess.summary}.` : "";
    el.lokrTrainingReadiness.textContent = `Preprocessing is running.${summary} Training will unlock when tensors are ready.`;
    el.trainLokrButton.disabled = true;
    return;
  }
  const readyRun = latestLokrPreprocessRun(datasetId);
  if (readyRun) {
    const summary = readyRun.summary || "Preprocess complete";
    el.lokrTrainingReadiness.textContent = `${summary}. Ready to train with tensors at ${readyRun.tensor_dir}.`;
    el.trainLokrButton.disabled = false;
    return;
  }
  const failedPreprocess = visibleLokrRuns().find((run) => run.dataset_id === datasetId && run.type === "preprocess" && run.status === "failed");
  if (failedPreprocess) {
    el.lokrTrainingReadiness.textContent = failedPreprocess.message || "Latest preprocess failed. View the run log, fix the dataset, then preprocess again.";
    el.trainLokrButton.disabled = true;
    return;
  }
  el.lokrTrainingReadiness.textContent = "Run preprocess to build the tensor dataset required for Side-Step training.";
  el.trainLokrButton.disabled = true;
}

function renderLokrRuns() {
  el.lokrRunList.replaceChildren();
  const activeRun = activeLokrRun();
  const runs = visibleLokrRuns();
  const running = runs.filter((run) => run.status === "running").length;
  const ready = activeLokrDataset() ? latestLokrPreprocessRun(activeLokrDataset().metadata.dataset_id) : null;
  setPill(el.lokrRunState, activeRun ? "Running" : ready ? "Ready to train" : running ? `${running} running` : `${runs.length} runs`, activeRun ? "warn" : ready ? "ok" : running ? "warn" : runs.length ? "ok" : "neutral");
  renderLokrTrainingReadiness();
  if (!runs.length) {
    const empty = document.createElement("div");
    empty.className = "empty-result";
    empty.textContent = state.lokrRunViewClearedAt ? "Run view cleared. New Side-Step runs will appear here." : "No Side-Step runs yet.";
    el.lokrRunList.appendChild(empty);
    return;
  }
  runs.slice(0, 12).forEach((run) => {
    const item = document.createElement("article");
    item.className = `generated-item lokr-run-item${run.run_id === state.selectedLokrRunId ? " active" : ""}`;
    const progress = lokrProgressDetails(run);
    item.innerHTML = `
      <div class="generated-title">
        <strong>${escapeHtml(run.label || run.run_id)}</strong>
        <span>${escapeHtml(run.status || "unknown")}</span>
      </div>
      ${progress.length ? `<div class="summary">${escapeHtml(progress.join(" | "))}</div>` : ""}
      ${run.summary ? `<div class="summary">${escapeHtml(run.summary)}${run.ready_to_train ? ". Ready to train." : ""}</div>` : ""}
      ${run.message ? `<div class="activity-readout"><strong>Message</strong><br>${escapeHtml(run.message)}</div>` : ""}
      <div class="asset-path">${escapeHtml((run.command || []).join(" "))}</div>
      <div class="button-row generated-actions">
        <button class="secondary-button lokr-view-log-button" type="button">View Log</button>
        ${run.status === "running" ? `<button class="secondary-button lokr-stop-run-button" type="button">Stop</button>` : ""}
      </div>
    `;
    item.querySelector(".lokr-view-log-button").addEventListener("click", () => loadLokrRunLog(run.run_id));
    const stopButton = item.querySelector(".lokr-stop-run-button");
    if (stopButton) stopButton.addEventListener("click", () => stopLokrRun(run.run_id));
    el.lokrRunList.appendChild(item);
  });
}

async function refreshLokrRuns() {
  const [runs, adapters] = await Promise.all([api("/api/lokr/runs"), api("/api/lokr/adapters")]);
  state.lokrRuns = runs;
  state.lokrAdapters = adapters;
  renderLokrRuns();
  renderMusicLokrAdapters();
}

async function loadLokrRunLog(runId) {
  state.selectedLokrRunId = runId;
  const response = await api(`/api/lokr/runs/${encodeURIComponent(runId)}/logs`);
  const run = state.lokrRuns.find((item) => item.run_id === runId);
  el.lokrRunLog.textContent = response.text || (run && run.message ? run.message : "No log output yet.");
  renderLokrRuns();
}

async function refreshSelectedLokrRunLog() {
  if (!state.selectedLokrRunId) return;
  const run = state.lokrRuns.find((item) => item.run_id === state.selectedLokrRunId);
  if (!run || run.status !== "running") return;
  const response = await api(`/api/lokr/runs/${encodeURIComponent(run.run_id)}/logs`);
  el.lokrRunLog.textContent = response.text || run.message || "No log output yet.";
}

async function stopLokrRun(runId = null) {
  const run = runId ? state.lokrRuns.find((item) => item.run_id === runId) : activeLokrRun();
  if (!run) {
    showToast("No running Side-Step run to stop");
    return;
  }
  el.stopLokrRunButton.disabled = true;
  try {
    const response = await api(`/api/lokr/runs/${encodeURIComponent(run.run_id)}/stop`, { method: "POST" });
    state.lokrRuns = state.lokrRuns.map((item) => item.run_id === response.run.run_id ? response.run : item);
    state.selectedLokrRunId = response.run.run_id;
    renderLokrRuns();
    await loadLokrRunLog(response.run.run_id);
    showToast(response.run.message || "Side-Step run stopped");
  } catch (error) {
    showToast(error.message || "Could not stop Side-Step run");
  } finally {
    renderLokrRuns();
  }
}

async function clearLokrLogView() {
  state.selectedLokrRunId = null;
  state.lokrRunViewClearedAt = Date.now();
  window.localStorage.setItem("danceStationLokrRunViewClearedAt", String(state.lokrRunViewClearedAt));
  el.lokrRunLog.textContent = "Select a run to view logs.";
  const logs = await api("/api/logs", { method: "DELETE" });
  renderLogs(logs);
  renderLokrRuns();
  showToast("Log view cleared");
}

async function preprocessLokrDataset() {
  if (activeLokrRun()) {
    showToast("Stop the current Side-Step run before starting another");
    return;
  }
  const dataset = activeLokrDataset();
  if (!dataset) {
    showToast("Select a LoKr dataset first");
    return;
  }
  const saved = await api(`/api/lokr/datasets/${encodeURIComponent(dataset.metadata.dataset_id)}`, {
    method: "POST",
    body: JSON.stringify({ dataset: lokrDatasetFromEditor() }),
  });
  setActiveLokrDataset(saved.dataset);
  const missingAudio = datasetMissingAudioCount(saved.dataset);
  if (missingAudio) {
    showToast(`Attach audio to ${missingAudio} entr${missingAudio === 1 ? "y" : "ies"} before preprocessing`);
    return;
  }
  const response = await api(`/api/lokr/datasets/${encodeURIComponent(dataset.metadata.dataset_id)}/preprocess`, {
    method: "POST",
    body: JSON.stringify(lokrRunPayload()),
  });
  state.lokrRuns.unshift(response.run);
  renderLokrRuns();
  state.selectedLokrRunId = response.run.run_id;
  await loadLokrRunLog(response.run.run_id);
  showToast(response.run.status === "failed" ? response.run.message || "Side-Step preprocess failed to start" : "Side-Step preprocess started");
}

async function trainLokrDataset() {
  if (activeLokrRun()) {
    showToast("Stop the current Side-Step run before starting another");
    return;
  }
  const dataset = activeLokrDataset();
  if (!dataset) {
    showToast("Select a LoKr dataset first");
    return;
  }
  if (datasetMissingAudioCount(dataset)) {
    showToast(`Attach audio to ${datasetMissingAudioCount(dataset)} entr${datasetMissingAudioCount(dataset) === 1 ? "y" : "ies"} before training`);
    return;
  }
  const response = await api(`/api/lokr/datasets/${encodeURIComponent(dataset.metadata.dataset_id)}/train`, {
    method: "POST",
    body: JSON.stringify(lokrTrainingPayload()),
  });
  state.lokrRuns.unshift(response.run);
  renderLokrRuns();
  state.selectedLokrRunId = response.run.run_id;
  await loadLokrRunLog(response.run.run_id);
  showToast(response.run.status === "failed" ? response.run.message || "Side-Step training failed to start" : "Side-Step LoKr training started");
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

async function openAssetInEditor(asset) {
  state.selectedEditorAsset = asset;
  const url = assetAudioUrl(asset);
  el.editorCurrentAsset.textContent = `${asset.category}: ${asset.label}`;
  el.editSourceAssetReadout.innerHTML = [
    `<strong>${escapeHtml(asset.label)}</strong>`,
    `Category: ${escapeHtml(asset.category)}`,
    `Path: ${escapeHtml(asset.audio_path)}`,
  ].join("<br>");
  if (!el.editSaveLabelInput.value.trim()) {
    el.editSaveLabelInput.value = `${asset.label} edit`;
  }
  try {
    await sendAudioBufferToEditor(url, asset.label);
    showToast("Loaded asset in Audio Editor");
  } catch (error) {
    showToast(error.message);
  }
}

function renderDatasetEditorSources() {
  el.datasetEditorTargetList.replaceChildren();
  el.datasetEditorDonorList.replaceChildren();
  setPill(el.datasetEditorTargetState, `${state.lokrDatasets.length} targets`, state.lokrDatasets.length ? "ok" : "neutral");
  setPill(el.datasetEditorDonorState, `${state.datasetSources.length} sources`, state.datasetSources.length ? "ok" : "neutral");

  if (!state.lokrDatasets.length) {
    const empty = document.createElement("div");
    empty.className = "empty-result";
    empty.textContent = "No target datasets yet.";
    el.datasetEditorTargetList.appendChild(empty);
  } else {
    state.lokrDatasets.forEach((dataset) => {
      const metadata = dataset.metadata || {};
      const row = document.createElement("article");
      row.className = `generated-item lokr-dataset-item${metadata.dataset_id === state.datasetEditorTargetId ? " active" : ""}`;
      row.innerHTML = `
        <div class="generated-title">
          <strong>${escapeHtml(metadata.label || metadata.name || "LoKr dataset")}</strong>
          <span>${Number(metadata.num_samples || (dataset.samples || []).length || 0)} samples</span>
        </div>
        <div class="asset-path">${escapeHtml(metadata.custom_tag ? `Trigger: ${metadata.custom_tag}` : "No trigger tag")}</div>
        <button class="secondary-button full-width" type="button">Edit Target</button>
      `;
      row.querySelector("button").addEventListener("click", async () => {
        const response = await api(`/api/lokr/datasets/${encodeURIComponent(metadata.dataset_id)}`);
        setActiveLokrDataset(response);
        state.datasetEditorTargetId = response.metadata.dataset_id;
        renderDatasetEditorSources();
        renderDatasetEditorTarget();
      });
      el.datasetEditorTargetList.appendChild(row);
    });
  }

  if (!state.datasetSources.length) {
    const empty = document.createElement("div");
    empty.className = "empty-result";
    empty.textContent = "No donor datasets available.";
    el.datasetEditorDonorList.appendChild(empty);
    return;
  }

  state.datasetSources.forEach((source) => {
    const summary = datasetSourceSummary(source);
    const row = document.createElement("article");
    row.className = `generated-item lokr-dataset-item${source.source_id === state.datasetEditorDonorSourceId ? " active" : ""}`;
    row.innerHTML = `
      <div class="generated-title">
        <strong>${escapeHtml(summary.label)}</strong>
        <span>${summary.count} samples</span>
      </div>
      <div class="asset-path">${escapeHtml(source.source_kind === "library" ? "Imported dataset" : "Local dataset")}</div>
      <button class="secondary-button full-width" type="button">Open Donor</button>
    `;
    row.querySelector("button").addEventListener("click", () => loadDatasetEditorDonor(source.source_id));
    el.datasetEditorDonorList.appendChild(row);
  });
}

function renderDatasetEditorTarget() {
  const dataset = activeDatasetEditorTarget();
  el.datasetEditorEntryList.replaceChildren();
  if (!dataset) {
    el.datasetEditorTargetReadout.textContent = "No target dataset selected";
    el.datasetEditorLabel.value = "";
    el.datasetEditorCustomTag.value = "";
    el.datasetEditorDefaultGenre.value = "";
    el.datasetEditorDefaultLanguage.value = "unknown";
    el.datasetEditorTagPosition.value = "prepend";
    el.datasetEditorGenreRatio.value = "0";
    el.datasetEditorSampleCount.value = "";
    el.datasetEditorAllInstrumental.checked = true;
    el.datasetEditorSaveButton.disabled = true;
    setPill(el.datasetEditorValidationState, "No dataset", "neutral");
    el.datasetEditorSummary.textContent = "Create or select a target dataset to begin editing.";
    const empty = document.createElement("div");
    empty.className = "empty-result";
    empty.textContent = "Create or select a target dataset.";
    el.datasetEditorEntryList.appendChild(empty);
    return;
  }

  const metadata = dataset.metadata || {};
  const samples = dataset.samples || [];
  const missingAudio = datasetMissingAudioCount(dataset);
  el.datasetEditorTargetReadout.textContent = metadata.dataset_id || "";
  el.datasetEditorLabel.value = metadata.label || metadata.name || "";
  el.datasetEditorCustomTag.value = metadata.custom_tag || "";
  el.datasetEditorDefaultGenre.value = metadata.default_genre || "";
  el.datasetEditorDefaultLanguage.value = metadata.default_language || "unknown";
  el.datasetEditorTagPosition.value = metadata.tag_position || "prepend";
  el.datasetEditorGenreRatio.value = String(metadata.genre_ratio ?? 0);
  el.datasetEditorSampleCount.value = `${samples.length}`;
  el.datasetEditorAllInstrumental.checked = Boolean(metadata.all_instrumental);
  el.datasetEditorSaveButton.disabled = false;

  const missingCaptions = samples.filter((sample) => !(sample.caption || "").trim()).length;
  setPill(
    el.datasetEditorValidationState,
    samples.length ? `${samples.length} samples` : "Empty",
    samples.length ? (missingCaptions || missingAudio ? "warn" : "ok") : "neutral",
  );
  el.datasetEditorSummary.innerHTML = [
    `<strong>${escapeHtml(metadata.label || "LoKr dataset")}</strong>`,
    `Samples: ${samples.length}`,
    `Missing audio: ${missingAudio}`,
    `Missing captions: ${missingCaptions}`,
    `JSON: ${escapeHtml(dataset.metadata_path || "")}`,
  ].join("<br>");

  if (!samples.length) {
    const empty = document.createElement("div");
    empty.className = "empty-result";
    empty.textContent = "No entries in this target dataset yet.";
    el.datasetEditorEntryList.appendChild(empty);
    return;
  }

  samples.forEach((sample, index) => {
    const item = document.createElement("article");
    item.className = "lokr-entry generated-item";
    item.dataset.entryId = sample.id;
    item.innerHTML = `
      <div class="generated-title">
        <strong>${escapeHtml(sample.label || sample.filename || `Sample ${index + 1}`)}</strong>
        <span>${sample.has_audio ? (sample.duration ? `${Number(sample.duration).toFixed(1)}s` : "duration unknown") : "audio missing"}</span>
      </div>
      ${sample.audio_url ? `<audio controls preload="metadata" src="${sample.audio_url}"></audio>` : ""}
      <div class="control-grid">
        <label class="field">
          <span>Label</span>
          <input class="dataset-editor-entry-label" type="text" value="${escapeHtml(sample.label || "")}" />
        </label>
        <label class="field">
          <span>Genre</span>
          <input class="dataset-editor-entry-genre" type="text" value="${escapeHtml(sample.genre || "")}" />
        </label>
        <label class="field">
          <span>Language</span>
          <input class="dataset-editor-entry-language" type="text" value="${escapeHtml(sample.language || "unknown")}" />
        </label>
        <label class="field">
          <span>BPM</span>
          <input class="dataset-editor-entry-bpm" type="text" value="${escapeHtml(String(sample.bpm ?? "N/A"))}" />
        </label>
        <label class="field">
          <span>Key</span>
          <input class="dataset-editor-entry-keyscale" type="text" value="${escapeHtml(sample.keyscale || "N/A")}" />
        </label>
        <label class="field">
          <span>Time signature</span>
          <input class="dataset-editor-entry-timesignature" type="text" value="${escapeHtml(sample.timesignature || "4")}" />
        </label>
      </div>
      <label class="field">
        <span>Caption</span>
        <textarea class="dataset-editor-entry-caption" rows="3">${escapeHtml(sample.caption || "")}</textarea>
      </label>
      <label class="field">
        <span>Lyrics</span>
        <textarea class="dataset-editor-entry-lyrics" rows="4">${escapeHtml(sample.lyrics || "[Instrumental]")}</textarea>
      </label>
      <div class="control-grid">
        <label class="field">
          <span>Trigger tag override</span>
          <input class="dataset-editor-entry-custom-tag" type="text" value="${escapeHtml(sample.custom_tag || "")}" />
        </label>
        <label class="field">
          <span>Prompt override</span>
          <select class="dataset-editor-entry-prompt-override">
            <option value="">Dataset default</option>
            <option value="caption"${sample.prompt_override === "caption" ? " selected" : ""}>Caption</option>
            <option value="genre"${sample.prompt_override === "genre" ? " selected" : ""}>Genre</option>
          </select>
        </label>
      </div>
      <div class="toggle-row">
        <label><input class="dataset-editor-entry-instrumental" type="checkbox"${sample.is_instrumental ? " checked" : ""} /> Instrumental</label>
        <label><input class="dataset-editor-entry-labeled" type="checkbox"${sample.labeled ? " checked" : ""} /> Labeled</label>
      </div>
      <div class="button-row generated-actions">
        <button class="secondary-button dataset-editor-attach-file-button" type="button">Attach File</button>
        <button class="secondary-button dataset-editor-delete-entry-button" type="button">Delete Entry</button>
      </div>
    `;
    item.querySelector(".dataset-editor-attach-file-button").addEventListener("click", () => attachFileToDatasetEditorEntry(sample.id));
    item.querySelector(".dataset-editor-delete-entry-button").addEventListener("click", () => deleteDatasetEditorTargetEntry(sample.id));
    el.datasetEditorEntryList.appendChild(item);
  });
}

function renderDatasetEditorDonor() {
  el.datasetEditorDonorEntryList.replaceChildren();
  const donor = state.datasetEditorDonor;
  if (!donor) {
    el.datasetEditorDonorReadout.textContent = "No donor selected";
    const empty = document.createElement("div");
    empty.className = "empty-result";
    empty.textContent = "Open a donor dataset to push entries into the target.";
    el.datasetEditorDonorEntryList.appendChild(empty);
    return;
  }
  const summary = datasetSourceSummary(donor);
  el.datasetEditorDonorReadout.textContent = `${summary.label} • ${summary.count} samples`;
  const samples = donor.samples || [];
  if (!samples.length) {
    const empty = document.createElement("div");
    empty.className = "empty-result";
    empty.textContent = "This donor dataset has no entries.";
    el.datasetEditorDonorEntryList.appendChild(empty);
    return;
  }
  samples.forEach((sample, index) => {
    const row = document.createElement("article");
    row.className = "generated-item";
    row.innerHTML = `
      <div class="generated-title">
        <strong>${escapeHtml(sample.label || sample.filename || `Sample ${index + 1}`)}</strong>
        <span>${sample.has_audio ? (sample.duration ? `${Number(sample.duration).toFixed(1)}s` : "duration unknown") : "audio missing"}</span>
      </div>
      ${sample.audio_url ? `<audio controls preload="metadata" src="${sample.audio_url}"></audio>` : ""}
      <div class="summary">
        ${escapeHtml(sample.genre || "No genre")}<br>
        ${escapeHtml((sample.caption || "").slice(0, 160) || "No caption")}
      </div>
      <div class="button-row generated-actions">
        <button class="primary-button dataset-editor-push-entry-button" type="button">Push To Target</button>
      </div>
    `;
    row.querySelector(".dataset-editor-push-entry-button").addEventListener("click", () => pushDonorEntryIntoTarget(sample));
    el.datasetEditorDonorEntryList.appendChild(row);
  });
}

function renameEndpointForAsset(asset) {
  if (asset.category === "transition") return `/api/transitions/${encodeURIComponent(asset.asset_id)}/rename`;
  if (asset.category === "generation") return `/api/music-generations/${encodeURIComponent(asset.asset_id)}/rename`;
  if (asset.category === "sound_effect") return `/api/sound-effects/${encodeURIComponent(asset.asset_id)}/rename`;
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
  renderRhythmAssetSelectors();
  renderVoiceWorkAssetOptions();
  renderVocal2BgmAssetOptions();
  renderVoiceWorkInputModes();
  renderVocal2BgmInputModes();
}

async function refreshLocalLibrary() {
  await refreshRhythmVolumes();
  const [response, connection] = await Promise.all([
    api("/api/library/local"),
    api("/api/library/publish/connection"),
  ]);
  state.localLibraryItems = response.items || [];
  state.localLibraryIndexPath = response.index_path || "";
  state.publicLibraryConnection = connection;
  renderLocalLibrary();
  renderRhythmAssetList();
  renderVoiceWorkAssetOptions();
  renderVoiceWorkInputModes();
  await refreshDatasetSources();
}

async function reindexLocalLibrary() {
  await refreshRhythmVolumes();
  setPill(el.libraryState, "Reindexing", "warn");
  const response = await api("/api/library/local/reindex", { method: "POST" });
  state.localLibraryItems = response.items || [];
  state.localLibraryIndexPath = response.index_path || "";
  renderLocalLibrary();
  renderRhythmAssetList();
  renderVoiceWorkAssetOptions();
  renderVoiceWorkInputModes();
  await refreshDatasetSources();
  showToast(`Indexed ${response.count || 0} local library items`);
}

async function saveLibraryItem(row, item) {
  const title = row.querySelector(".library-title-input").value.trim();
  const description = row.querySelector(".library-description-input").value.trim();
  const tags = row
    .querySelector(".library-tags-input")
    .value.split(",")
    .map((tag) => tag.trim())
    .filter(Boolean);
  if (!title) {
    showToast("Enter a title");
    return;
  }
  try {
    const response = await api(`/api/library/local/${encodeURIComponent(item.id)}`, {
      method: "PATCH",
      body: JSON.stringify({ title, description, tags }),
    });
    const index = state.localLibraryItems.findIndex((candidate) => candidate.id === item.id);
    if (index >= 0) state.localLibraryItems[index] = response.item;
    renderLocalLibrary();
    showToast("Library metadata saved");
  } catch (error) {
    showToast(error.message);
  }
}

async function setLibraryCardImage(item) {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = "image/png,image/jpeg,image/webp,image/gif";
  input.addEventListener("change", async () => {
    const file = input.files && input.files[0];
    if (!file) return;
    const form = new FormData();
    form.set("file", file, file.name);
    try {
      const response = await fetch(`/api/library/local/${encodeURIComponent(item.id)}/cover`, {
        method: "POST",
        body: form,
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.detail || payload.error || "Could not set card image");
      }
      const index = state.localLibraryItems.findIndex((candidate) => candidate.id === item.id);
      if (index >= 0) state.localLibraryItems[index] = payload.item;
      renderLocalLibrary();
      showToast("Card image saved");
    } catch (error) {
      showToast(error.message);
    }
  });
  input.click();
}

async function connectLibraryWallet() {
  const wallet = el.libraryWalletProvider.value || "phantom";
  const provider = getWalletProvider(wallet);
  if (!provider) {
    showToast(`${wallet} wallet is not available in this browser`);
    renderLibraryConnection();
    return;
  }
  try {
    setPreferredLibraryWallet(wallet);
    setPill(el.libraryPublishState, "Connecting", "warn");
    const savedConnection = await api("/api/library/publish/connection", {
      method: "POST",
      body: JSON.stringify({}),
    });
    state.publicLibraryConnection = savedConnection;
    const connected = await provider.connect();
    const publicKey = connected && connected.publicKey && connected.publicKey.toString ? connected.publicKey.toString() : "";
    if (!publicKey) {
      throw new Error("Wallet did not provide a public key");
    }
    const nonce = await api("/api/library/publish/auth/nonce", {
      method: "POST",
      body: JSON.stringify({ public_key: publicKey }),
    });
    const messageBytes = new TextEncoder().encode(nonce.message);
    const signed = await provider.signMessage(messageBytes, "utf8");
    const signature = Array.from(signed.signature || []);
    const connection = await api("/api/library/publish/auth/verify", {
      method: "POST",
      body: JSON.stringify({
        public_key: publicKey,
        nonce: nonce.nonce,
        message: nonce.message,
        signature,
      }),
    });
    state.publicLibraryConnection = connection;
    renderLibraryConnection();
    renderLocalLibrary();
    showToast("Wallet connected to public library");
  } catch (error) {
    renderLibraryConnection();
    showToast(error.message);
  }
}

async function disconnectLibraryWallet() {
  try {
    const connection = await api("/api/library/publish/auth/logout", { method: "POST" });
    state.publicLibraryConnection = connection;
    renderLibraryConnection();
    renderLocalLibrary();
    showToast("Public library wallet disconnected");
  } catch (error) {
    showToast(error.message);
  }
}

async function refreshPublicLibrary() {
  const kind = el.publicLibraryKind.value || "all";
  setPill(el.publicLibraryState, "Loading", "warn");
  const response = await api(`/api/library/public?kind=${encodeURIComponent(kind)}`);
  state.publicLibraryItems = response.items || [];
  renderPublicLibrary();
}

async function importPublicLibraryItem(row, item) {
  const button = row.querySelector(".public-import-button");
  try {
    button.disabled = true;
    button.textContent = "Importing...";
    setPill(el.publicLibraryState, "Importing", "warn");
    const response = await api(`/api/library/public/${encodeURIComponent(item.id)}/import`, { method: "POST" });
    const existingIndex = state.localLibraryItems.findIndex((candidate) => candidate.id === response.item.id);
    if (existingIndex >= 0) {
      state.localLibraryItems[existingIndex] = response.item;
    } else {
      state.localLibraryItems.unshift(response.item);
    }
    await refreshEditorAssets();
    await refreshDatasetSources();
    renderLocalLibrary();
    renderPublicLibrary();
    showToast("Imported public library item");
  } catch (error) {
    showToast(error.message);
    setPill(el.publicLibraryState, "Import failed", "error");
  } finally {
    button.disabled = false;
    button.textContent = "Import";
  }
}

async function publishLibraryItem(row, item) {
  const button = row.querySelector(".library-publish-button");
  state.publicLibraryConnection = await api("/api/library/publish/connection");
  renderLibraryConnection();
  renderLocalLibrary();
  if (!state.publicLibraryConnection?.authenticated) {
    showToast("Connect your wallet to publish");
    return;
  }
  try {
    const publish = (item.metadata || {}).public_library || null;
    const isPublished = Boolean(publish && publish.remote_status === "published" && publish.remote_visibility === "public");
    state.libraryPublishingItemIds.add(item.id);
    renderLocalLibrary();
    button.disabled = true;
    button.textContent = isPublished ? "Updating..." : "Publishing...";
    setPill(el.libraryPublishState, isPublished ? "Updating" : "Publishing", "warn");
    const response = await api(`/api/library/local/${encodeURIComponent(item.id)}/publish`, {
      method: "POST",
      body: JSON.stringify({ publish_public: true }),
    });
    const index = state.localLibraryItems.findIndex((candidate) => candidate.id === item.id);
    if (index >= 0) state.localLibraryItems[index] = response.item;
    renderLocalLibrary();
    showToast(`${isPublished ? "Updated" : "Published"} ${response.publish.file_count || 0} files`);
  } catch (error) {
    showToast(error.message);
    setPill(el.libraryPublishState, "Publish failed", "error");
  } finally {
    state.libraryPublishingItemIds.delete(item.id);
    renderLocalLibrary();
    button.disabled = false;
    button.textContent = "Publish";
  }
}

async function revokeLibraryItem(row, item) {
  const button = row.querySelector(".library-revoke-button");
  if (!button) return;
  state.publicLibraryConnection = await api("/api/library/publish/connection");
  renderLibraryConnection();
  renderLocalLibrary();
  if (!state.publicLibraryConnection?.authenticated) {
    showToast("Connect your wallet to revoke");
    return;
  }
  try {
    state.libraryRevokingItemIds.add(item.id);
    renderLocalLibrary();
    button.disabled = true;
    button.textContent = "Revoking...";
    setPill(el.libraryPublishState, "Revoking", "warn");
    const response = await api(`/api/library/local/${encodeURIComponent(item.id)}/revoke`, {
      method: "POST",
    });
    const index = state.localLibraryItems.findIndex((candidate) => candidate.id === item.id);
    if (index >= 0) state.localLibraryItems[index] = response.item;
    renderLocalLibrary();
    showToast("Public asset revoked");
  } catch (error) {
    showToast(error.message);
    setPill(el.libraryPublishState, "Revoke failed", "error");
  } finally {
    state.libraryRevokingItemIds.delete(item.id);
    renderLocalLibrary();
    button.disabled = false;
    button.textContent = "Revoke";
  }
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
  if (["downloading", "initializing", "generating", "recovering"].includes(phase)) return "warn";
  if (phase === "ready" || phase === "complete") return "ok";
  return "neutral";
}

function activityLabel(activity) {
  const phase = activity.phase || "idle";
  if (phase === "recovering") return "Recovering";
  if (phase === "downloading") return "Downloading";
  if (phase === "initializing") return "Initializing";
  if (phase === "generating") return "Generating";
  if (phase === "error") return "Runtime error";
  if (phase === "ready") return "Runtime ready";
  return "Waiting";
}

function renderActivity(activity) {
  state.runtimeActivity = activity;
  const message = activity.message || "No ACE-Step activity yet.";
  const detail = activity.detail ? `<br>${activity.detail}` : "";
  el.generationActivity.innerHTML = `<strong>${activityLabel(activity)}</strong><br>${message}${detail}`;
  if (aceRuntimeBusy()) {
    setPill(el.actionState, "Recovering", "warn");
  } else if (state.isGenerating) {
    setPill(el.actionState, activityLabel(activity), activityTone(activity.phase));
  }
}

async function refreshActivity() {
  const activity = await api("/api/runtime/activity");
  renderActivity(activity);
  return activity;
}

async function refreshRuntimeState() {
  const runtime = await api("/api/runtime/status");
  renderRuntime(runtime);
  return runtime;
}

function startRuntimeRecoveryPolling() {
  if (state.runtimeRecoveryPollTimer) return;
  state.runtimeRecoveryPollTimer = window.setInterval(async () => {
    try {
      const [runtime] = await Promise.all([refreshRuntimeState(), refreshActivity(), refreshLogs(), refreshExtractions()]);
      if (state.activeRhythmProjectId) {
        await loadRhythmProject(state.activeRhythmProjectId, false).catch(() => {});
      } else {
        await refreshRhythmProjects().catch(() => {});
      }
      if (!(runtime.recovery && runtime.recovery.active)) {
        const settledLabel = runtime.api_running ? "Ready" : "Error";
        const settledTone = runtime.api_running ? "ok" : "error";
        if (el.extractActionState && el.extractActionState.textContent === "Recovering") {
          setPill(el.extractActionState, settledLabel, settledTone);
        }
        if (el.musicActionState && el.musicActionState.textContent === "Recovering") {
          setPill(el.musicActionState, settledLabel, settledTone);
        }
        if (el.rhythmAnalysisState && el.rhythmAnalysisState.textContent === "Recovering") {
          setPill(el.rhythmAnalysisState, settledLabel, settledTone);
        }
        window.clearInterval(state.runtimeRecoveryPollTimer);
        state.runtimeRecoveryPollTimer = null;
      }
    } catch (error) {
      // Keep polling until the runtime manager answers again.
    }
  }, 2500);
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
    const adapter = item.lokr_adapter || null;
    const adapterLabel = adapter ? `${adapter.label || adapter.adapter_id || "LoKr"} (${adapter.model || "model"})` : "None";
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
        <dt>LoKr</dt><dd>${escapeHtml(adapterLabel)}</dd>
        <dt>Prompt</dt><dd>${escapeHtml(item.prompt || "")}</dd>
        <dt>Output</dt><dd>${escapeHtml(outputPath || "None")}</dd>
        <dt>Metadata</dt><dd>${escapeHtml(item.metadata_path || "")}</dd>
      </dl>
    `;
    el.musicList.appendChild(row);
  });
}

function renderSoundEffectGenerations() {
  if (!el.soundEffectList) return;
  el.soundEffectList.replaceChildren();
  if (!state.soundEffectResults.length) {
    const empty = document.createElement("div");
    empty.className = "empty-result";
    empty.textContent = "No sound effects yet.";
    el.soundEffectList.appendChild(empty);
    return;
  }

  state.soundEffectResults.forEach((item, index) => {
    const row = document.createElement("article");
    row.className = "generated-item";
    const outputPath = item.generated_audio_path || "";
    const audio = outputPath
      ? `<audio controls preload="metadata" src="/api/sound-effects/audio?path=${encodeURIComponent(outputPath)}"></audio>`
      : `<div class="empty-result">No playable audio for this generation.</div>`;
    row.innerHTML = `
      <div class="generated-title">
        <strong>${index === 0 ? "Latest" : "Sound Effect"} - ${escapeHtml(item.status)}</strong>
        <span>${escapeHtml(item.label || item.model || "sound effect")}</span>
      </div>
      ${audio}
      <dl class="path-list">
        <dt>Message</dt><dd>${escapeHtml(item.message || "")}</dd>
        <dt>Prompt</dt><dd>${escapeHtml(item.prompt || "")}</dd>
        <dt>Duration</dt><dd>${escapeHtml(String(item.duration_seconds || ""))}</dd>
        <dt>Steps</dt><dd>${escapeHtml(String(item.steps || ""))}</dd>
        <dt>Output</dt><dd>${escapeHtml(outputPath || "None")}</dd>
        <dt>Metadata</dt><dd>${escapeHtml(item.metadata_path || "")}</dd>
      </dl>
    `;
    el.soundEffectList.appendChild(row);
  });
}

async function refreshSoundEffectGenerations() {
  state.soundEffectResults = await api("/api/sound-effects");
  renderSoundEffectGenerations();
}

async function runSoundEffectGeneration() {
  const label = el.soundEffectLabelInput ? el.soundEffectLabelInput.value.trim() : "";
  const prompt = el.soundEffectPromptInput ? el.soundEffectPromptInput.value.trim() : "";
  if (!label) {
    showToast("Enter a result label");
    return;
  }
  if (!prompt) {
    showToast("Enter a prompt");
    return;
  }
  if (el.runSoundEffectButton) el.runSoundEffectButton.disabled = true;
  setPill(el.soundEffectActionState, "Generating", "warn");
  if (el.soundEffectActivity) {
    el.soundEffectActivity.innerHTML = "<strong>Starting</strong><br>Preparing TangoFlux sound effect request.";
  }
  try {
    const response = await api("/api/sound-effects/run", {
      method: "POST",
      body: JSON.stringify({
        label,
        prompt,
        duration_seconds: numericValue(el.soundEffectDurationInput) ?? 10,
        steps: numericValue(el.soundEffectStepsInput) ?? 50,
        output_format: el.soundEffectOutputFormat ? el.soundEffectOutputFormat.value : "wav",
      }),
    });
    state.soundEffectResults.unshift(response.generation);
    state.soundEffectResults = state.soundEffectResults.slice(0, 24);
    renderSoundEffectGenerations();
    await refreshLocalLibrary();
    if (response.generation.status === "complete") {
      setPill(el.soundEffectActionState, "Complete", "ok");
      if (el.soundEffectActivity) {
        el.soundEffectActivity.innerHTML = "<strong>Complete</strong><br>Sound effect generation finished.";
      }
    } else {
      setPill(el.soundEffectActionState, "Failed", "error");
      if (el.soundEffectActivity) {
        el.soundEffectActivity.innerHTML = `<strong>Failed</strong><br>${escapeHtml(response.generation.message)}`;
      }
    }
    showToast(response.generation.message);
    await refreshLogs();
  } catch (error) {
    setPill(el.soundEffectActionState, "Error", "error");
    if (el.soundEffectActivity) {
      el.soundEffectActivity.innerHTML = `<strong>Error</strong><br>${escapeHtml(error.message)}`;
    }
    showToast(error.message);
  } finally {
    if (el.runSoundEffectButton) el.runSoundEffectButton.disabled = false;
  }
}

function renderVocal2BgmAssetOptions() {
  const current = el.vocal2bgmSourceAssetSelect ? el.vocal2bgmSourceAssetSelect.value : "";
  const assets = voiceWorkAssetEntries();
  if (!el.vocal2bgmSourceAssetSelect) return;
  el.vocal2bgmSourceAssetSelect.replaceChildren();
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = assets.length ? "Choose an existing creation" : "No audio assets available";
  el.vocal2bgmSourceAssetSelect.appendChild(placeholder);
  assets.forEach((asset) => {
    const option = document.createElement("option");
    option.value = asset.asset_id || "";
    option.textContent = `${asset.label || asset.asset_id || "asset"}${asset.category ? ` (${asset.category})` : ""}`;
    el.vocal2bgmSourceAssetSelect.appendChild(option);
  });
  if (current && [...el.vocal2bgmSourceAssetSelect.options].some((option) => option.value === current)) {
    el.vocal2bgmSourceAssetSelect.value = current;
  }
  const selected = selectedSourceAsset(el.vocal2bgmSourceAssetSelect);
  if (el.vocal2bgmSourceAssetName) {
    el.vocal2bgmSourceAssetName.textContent = selected
      ? `${selected.category}: ${selected.label}`
      : "No creation selected";
  }
}

function renderVocal2BgmInputModes() {
  const upload = el.vocal2bgmSourceUploadMode ? el.vocal2bgmSourceUploadMode.checked : true;
  if (el.vocal2bgmSourceUploadBlock) el.vocal2bgmSourceUploadBlock.classList.toggle("is-hidden", !upload);
  if (el.vocal2bgmSourceAssetBlock) el.vocal2bgmSourceAssetBlock.classList.toggle("is-hidden", upload);
  state.vocal2bgmSourceInputMode = upload ? "upload" : "asset";
  const hasUploadFile = Boolean(el.vocal2bgmSourceFile && el.vocal2bgmSourceFile.files && el.vocal2bgmSourceFile.files.length);
  const hasSelectedAsset = Boolean(selectedSourceAsset(el.vocal2bgmSourceAssetSelect));
  if ((upload && !hasUploadFile) || (!upload && !hasSelectedAsset)) {
    state.vocal2bgmSourcePath = "";
    state.vocal2bgmSourceProbe = null;
  }
  if (el.vocal2bgmSourceReadout) el.vocal2bgmSourceReadout.textContent = vocal2bgmSourceReadout();
}

function vocal2bgmSourceReadout() {
  const path = state.vocal2bgmSourcePath || "";
  const probe = state.vocal2bgmSourceProbe || null;
  if (!path) return "No source audio loaded.";
  const duration = probe ? formatTime(probe.duration_seconds) : "unknown";
  const format = probe ? probe.source_format : "unknown";
  return `Source ready: ${path} (${format}, ${duration})`;
}

async function loadVocal2BgmSourcePath(sourcePath, probe = null) {
  state.vocal2bgmSourcePath = sourcePath;
  state.vocal2bgmSourceProbe = probe;
  if (el.vocal2bgmSourceReadout) el.vocal2bgmSourceReadout.textContent = vocal2bgmSourceReadout();
}

async function probeVocal2BgmSource(sourcePath) {
  const probe = await api("/api/source/probe", {
    method: "POST",
    body: JSON.stringify({ source_path: sourcePath }),
  });
  await loadVocal2BgmSourcePath(sourcePath, probe);
}

async function uploadVocal2BgmSourceFile() {
  const file = el.vocal2bgmSourceFile.files && el.vocal2bgmSourceFile.files[0];
  if (!file) return;
  setPill(el.vocal2bgmActionState, "Loading", "warn");
  el.vocal2bgmSourceFileName.textContent = file.name;
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
    if (el.vocal2bgmSourceUploadMode) el.vocal2bgmSourceUploadMode.checked = true;
    if (el.vocal2bgmSourceAssetMode) el.vocal2bgmSourceAssetMode.checked = false;
    if (el.vocal2bgmSourceAssetSelect) el.vocal2bgmSourceAssetSelect.value = "";
    if (el.vocal2bgmSourceAssetName) el.vocal2bgmSourceAssetName.textContent = "No creation selected";
    renderVocal2BgmInputModes();
    await loadVocal2BgmSourcePath(body.stored_path, body.probe);
    setPill(el.vocal2bgmActionState, "Ready", "ok");
    showToast("Vocal2BGM source loaded");
  } catch (error) {
    setPill(el.vocal2bgmActionState, "Error", "error");
    showToast(error.message);
  } finally {
    refreshLogs();
  }
}

function renderVocal2BgmGenerations() {
  if (!el.vocal2bgmList) return;
  el.vocal2bgmList.replaceChildren();
  if (!state.vocal2bgmResults.length) {
    const empty = document.createElement("div");
    empty.className = "empty-result";
    empty.textContent = "No Vocal2BGM generations yet.";
    el.vocal2bgmList.appendChild(empty);
    return;
  }
  state.vocal2bgmResults.forEach((item, index) => {
    const row = document.createElement("article");
    row.className = "generated-item";
    const outputPath = item.generated_audio_path || "";
    const sourcePath = item.source_audio_path || "";
    const audio = outputPath
      ? `<audio controls preload="metadata" src="/api/music-generations/audio?path=${encodeURIComponent(outputPath)}"></audio>`
      : `<div class="empty-result">No playable audio for this generation.</div>`;
    row.innerHTML = `
      <div class="generated-title">
        <strong>${index === 0 ? "Latest" : "Vocal2BGM"} - ${escapeHtml(item.status || "complete")}</strong>
        <span>${escapeHtml(item.label || item.generation_id || "Vocal2BGM")}</span>
      </div>
      ${audio}
      <dl class="path-list">
        <dt>Message</dt><dd>${escapeHtml(item.message || "")}</dd>
        <dt>Model</dt><dd>${escapeHtml(item.model || "acestep-v15-base")}</dd>
        <dt>Source</dt><dd>${escapeHtml(sourcePath)}</dd>
        <dt>Duration</dt><dd>${escapeHtml(item.source_duration_seconds ? formatTime(item.source_duration_seconds) : "")}</dd>
        <dt>Output</dt><dd>${escapeHtml(outputPath || "None")}</dd>
        <dt>Metadata</dt><dd>${escapeHtml(item.metadata_path || "")}</dd>
      </dl>
    `;
    el.vocal2bgmList.appendChild(row);
  });
}

async function runVocal2BgmGeneration() {
  const label = el.vocal2bgmLabelInput.value.trim();
  const prompt = el.vocal2bgmPromptInput ? el.vocal2bgmPromptInput.value.trim() : "";
  if (!label) {
    showToast("Enter a result label");
    return;
  }
  const sourcePath = state.vocal2bgmSourcePath || "";
  if (!sourcePath) {
    showToast("Choose a source vocal file or creation");
    return;
  }
  setPill(el.vocal2bgmActionState, "Generating", "warn");
  el.vocal2bgmActivity.innerHTML = "<strong>Starting</strong><br>Preparing ACE-Step Base Vocal2BGM request.";
  el.runVocal2BgmButton.disabled = true;
  try {
    const response = await api("/api/music-generations/vocal2bgm", {
      method: "POST",
      body: JSON.stringify({
        source_audio_path: sourcePath,
        label,
        prompt,
        output_format: el.vocal2bgmOutputFormat.value,
        audio_duration:
          state.vocal2bgmSourceProbe && Number(state.vocal2bgmSourceProbe.duration_seconds || 0) > 0
            ? Number(state.vocal2bgmSourceProbe.duration_seconds)
            : null,
        inference_steps: numericValue(el.vocal2bgmInferenceSteps),
        guidance_scale: numericValue(el.vocal2bgmGuidanceScale),
        shift: numericValue(el.vocal2bgmShift),
        infer_method: el.vocal2bgmInferMethod.value,
        use_tiled_decode: el.vocal2bgmUseTiledDecode.checked,
        dcw_enabled: el.vocal2bgmDcwEnabled.checked,
        velocity_norm_threshold: 0,
        velocity_ema_factor: 0,
        seed: numericValue(el.vocal2bgmSeed),
        audio_cover_strength: numericValue(el.vocal2bgmSourceStrength) ?? 1,
      }),
    });
    state.vocal2bgmResults.unshift(response.generation);
    state.vocal2bgmResults = state.vocal2bgmResults.slice(0, 24);
    renderVocal2BgmGenerations();
    await refreshEditorAssets();
    if (response.generation.status === "complete") {
      setPill(el.vocal2bgmActionState, "Complete", "ok");
      el.vocal2bgmActivity.innerHTML = "<strong>Complete</strong><br>Vocal2BGM generation finished.";
    } else if (response.generation.status === "recovering") {
      setPill(el.vocal2bgmActionState, "Recovering", "warn");
      el.vocal2bgmActivity.innerHTML = `<strong>Recovering</strong><br>${escapeHtml(response.generation.message)}`;
      startRuntimeRecoveryPolling();
    } else {
      setPill(el.vocal2bgmActionState, "Failed", "error");
      el.vocal2bgmActivity.innerHTML = `<strong>Failed</strong><br>${escapeHtml(response.generation.message)}`;
    }
    showToast(response.generation.message);
  } catch (error) {
    setPill(el.vocal2bgmActionState, "Error", "error");
    el.vocal2bgmActivity.innerHTML = `<strong>Error</strong><br>${escapeHtml(error.message)}`;
    showToast(error.message);
  } finally {
    el.runVocal2BgmButton.disabled = false;
    refreshLogs();
  }
}

function selectedVoiceWorkVoice() {
  return state.voiceVoices.find((voice) => voice.voice_id === state.selectedVoiceWorkVoiceId) || null;
}

function renderVoiceWorkRuntime() {
  const runtime = state.voiceWorkStatus || null;
  if (!runtime) {
    if (el.voiceWorkRuntimeState) setPill(el.voiceWorkRuntimeState, "Unknown", "neutral");
    if (el.voiceWorkRuntimeDetails) el.voiceWorkRuntimeDetails.textContent = "Seed-VC runtime status not loaded.";
    setPill(el.voiceWorkRuntimeActionState, "Idle", "neutral");
    el.voiceWorkRuntimeActionDetails.textContent = "Install the runtime once, then start it here.";
    applyVoiceWorkAvailability();
    return;
  }
  const runtimePhase = voiceRuntimePhaseDisplay(runtime.phase);
  if (el.voiceWorkRuntimeState) setPill(el.voiceWorkRuntimeState, runtimePhase.label, runtimePhase.tone);
  if (el.voiceWorkRuntimeDetails) {
    const jobDetails = runtime.job && runtime.job.details ? runtime.job.details : {};
    const requestId = jobDetails && jobDetails.request_id ? String(jobDetails.request_id) : "";
    el.voiceWorkRuntimeDetails.innerHTML = [
      `<strong>${escapeHtml(runtime.phase_message || runtime.message || "")}</strong>`,
      runtime.startup_progress && runtime.startup_progress.phase && runtime.startup_progress.phase !== "idle"
        ? `Startup: ${escapeHtml(runtime.startup_progress.phase)}`
        : "",
      `Install dir: ${escapeHtml(runtime.install_dir || "")}`,
      `API: ${escapeHtml(runtime.api_url || "")}`,
      `Managed pid: ${runtime.managed_pid || "none"}${runtime.managed_pid_alive ? " (alive)" : ""}`,
      `Install: ${escapeHtml(runtime.install_command || "")}`,
      `Start: ${escapeHtml(runtime.start_command || "")}`,
      runtime.job && runtime.job.active
        ? `Job: ${escapeHtml(runtime.job.action || "working")} - ${escapeHtml(runtime.job.message || "")}${requestId ? `<br>Request: ${escapeHtml(requestId)}` : ""}`
        : runtime.job && runtime.job.phase === "failed"
          ? `Job failed: ${escapeHtml(runtime.job.action || "working")} - ${escapeHtml(runtime.job.error || runtime.job.message || "Unknown error")}${requestId ? `<br>Request: ${escapeHtml(requestId)}` : ""}`
        : runtime.job && runtime.job.completed_at
          ? `Job complete: ${escapeHtml(runtime.job.action || "working")} - ${escapeHtml(runtime.job.message || "")}${requestId ? `<br>Request: ${escapeHtml(requestId)}` : ""}`
            : "",
    ].join("<br>");
  }

  const action = runtime.action || null;
  const actionPhase = voiceRuntimePhaseDisplay(action && action.phase ? action.phase : "idle");
  const actionLabel = action && action.active ? actionPhase.label : action && action.phase === "failed" ? "Failed" : "Idle";
  const actionTone = action && action.active ? actionPhase.tone : action && action.phase === "failed" ? "error" : "neutral";
  setPill(el.voiceWorkRuntimeActionState, actionLabel, actionTone);
  el.voiceWorkRuntimeActionDetails.textContent =
    action && action.active
      ? action.message || "Seed-VC runtime action in progress."
    : action && action.phase === "failed"
      ? action.error || action.message || "Seed-VC runtime action failed."
      : runtime.phase === "ready"
        ? "Runtime is ready. Seed-VC uses the backend service directly."
        : runtime.phase === "starting"
          ? runtime.phase_message || "Seed-VC runtime is starting."
          : runtime.phase === "stale"
            ? runtime.phase_message || "Seed-VC process exists, but the UI is not reachable yet."
            : "Install the runtime once, then start it here.";

  const actionActive = Boolean(action && action.active);
  const runtimeBooting = runtime.phase === "starting" || runtime.phase === "stale";
  el.startVoiceWorkRuntimeButton.textContent = runtime.simple_start_command || "Start Runtime";
  if (el.installVoiceWorkRuntimeButton) el.installVoiceWorkRuntimeButton.textContent = runtime.simple_setup_command || "Install Runtime";
  el.installVoiceWorkRuntimeButton.disabled = actionActive;
  el.startVoiceWorkRuntimeButton.disabled = actionActive || runtimeBooting || runtime.phase === "missing";
  if (el.stopVoiceWorkRuntimeButton) el.stopVoiceWorkRuntimeButton.disabled = actionActive || (!runtime.api_running && !runtime.managed_pid_alive);
  if (runtime.job && runtime.job.active) {
    if (runtime.job.action === "convert") {
      if (el.voiceWorkGenerateState) setPill(el.voiceWorkGenerateState, "Converting", "warn");
      if (el.voiceWorkTrainingState) setPill(el.voiceWorkTrainingState, "Converting", "warn");
    }
  } else if (runtime.job && runtime.job.phase === "failed") {
    if (el.voiceWorkGenerateState) setPill(el.voiceWorkGenerateState, "Failed", "error");
    if (el.voiceWorkTrainingState) setPill(el.voiceWorkTrainingState, "Failed", "error");
  } else if (runtime.job && runtime.job.completed_at) {
    if (runtime.job.action === "convert") {
      if (el.voiceWorkGenerateState) setPill(el.voiceWorkGenerateState, "Complete", "ok");
      if (el.voiceWorkTrainingState) setPill(el.voiceWorkTrainingState, "Complete", "ok");
    }
  } else {
    if (el.voiceWorkGenerateState) setPill(el.voiceWorkGenerateState, "Ready", "neutral");
    if (el.voiceWorkTrainingState) {
      setPill(el.voiceWorkTrainingState, runtime.job && runtime.job.phase === "failed" ? "Failed" : "Idle", runtime.job && runtime.job.phase === "failed" ? "error" : "neutral");
    }
  }
  applyVoiceWorkAvailability();
}

function renderVoiceWorkSelection() {
  const voice = selectedVoiceWorkVoice();
  el.voiceWorkSelectedVoice.textContent = voice
    ? `Selected target voice: ${voice.label || voice.voice_id}${voice.language ? ` (${voice.language})` : ""}`
    : "No target voice selected.";
  if (el.voiceWorkSampleVoiceSelect) {
    el.voiceWorkSampleVoiceSelect.value = voice ? voice.voice_id : "";
  }
}

function populateVoiceWorkVoiceInputs(voice) {
  if (!voice) return;
  el.voiceWorkLabel.value = voice.label || "";
  el.voiceWorkLanguage.value = voice.language || "auto";
  el.voiceWorkDescription.value = voice.description || "";
}

function voiceWorkAssetEntries() {
  const combined = new Map();
  const push = (asset, fallbackKind) => {
    if (!asset) return;
    const files = Array.isArray(asset.files) ? asset.files : [];
    const fileAudioPath = files.find((file) => file && typeof file.path === "string" && looksLikePlayableAudio(file.path));
    const audioPath =
      asset.audio_path ||
      asset.generated_audio_path ||
      asset.output_audio_path ||
      asset.preview_audio_path ||
      (fileAudioPath && fileAudioPath.path) ||
      "";
    if (!audioPath || !looksLikePlayableAudio(audioPath)) return;
    const category = asset.category || asset.kind || fallbackKind || "asset";
    if (category === "dataset" || category === "metadata") return;
    const assetId = String(asset.asset_id || asset.id || asset.voice_id || asset.library_item_id || audioPath);
    const dedupeKey = `${audioPath}::${category}`;
    if (combined.has(dedupeKey)) return;
    const filename = String(audioPath).split(/[\\/]/).pop() || "";
    const label = asset.label || asset.title || asset.name || asset.display_name || asset.voice_label || filename.replace(/\.[^.]+$/, "");
    combined.set(dedupeKey, {
      asset_id: assetId,
      label,
      category,
      audio_path: audioPath,
      audio_url: asset.audio_url || asset.audioUrl || "",
      kind: asset.kind || "",
    });
  };
  for (const asset of state.editorAssets || []) {
    push(asset, asset && asset.category);
  }
  for (const item of state.localLibraryItems || []) {
    push(item, item && item.kind);
  }
  return Array.from(combined.values());
}

function renderVoiceWorkAssetOptions() {
  const currentTarget = el.voiceWorkAssetSelect ? el.voiceWorkAssetSelect.value : "";
  const currentSource = el.voiceWorkSourceAssetSelect ? el.voiceWorkSourceAssetSelect.value : "";
  const assets = voiceWorkAssetEntries();
  const populate = (select, current) => {
    if (!select) return;
    select.replaceChildren();
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = assets.length ? "Choose an existing creation" : "No audio assets available";
    select.appendChild(placeholder);
    assets.forEach((asset) => {
      const option = document.createElement("option");
      option.value = asset.asset_id || asset.id || "";
      option.textContent = `${asset.label || asset.title || asset.asset_id || "asset"}${asset.category ? ` (${asset.category})` : ""}`;
      select.appendChild(option);
    });
    if (current && [...select.options].some((option) => option.value === current)) {
      select.value = current;
    }
  };
  populate(el.voiceWorkAssetSelect, currentTarget);
  populate(el.voiceWorkSourceAssetSelect, currentSource);
  if (el.voiceWorkAssetName) {
    const selected = selectedVoiceWorkAsset(el.voiceWorkAssetSelect);
    el.voiceWorkAssetName.textContent = selected
      ? `${selected.category}: ${selected.label}`
      : "No creation selected";
  }
  if (el.voiceWorkSourceAssetName) {
    const selected = selectedSourceAsset(el.voiceWorkSourceAssetSelect);
    el.voiceWorkSourceAssetName.textContent = selected
      ? `${selected.category}: ${selected.label}`
      : "No source asset selected";
  }
}

function renderVoiceWorkInputModes() {
  const targetUpload = el.voiceWorkTargetUploadMode ? el.voiceWorkTargetUploadMode.checked : true;
  const sourceUpload = el.voiceWorkSourceUploadMode ? el.voiceWorkSourceUploadMode.checked : true;
  if (el.voiceWorkTargetUploadBlock) el.voiceWorkTargetUploadBlock.classList.toggle("is-hidden", !targetUpload);
  if (el.voiceWorkTargetAssetBlock) el.voiceWorkTargetAssetBlock.classList.toggle("is-hidden", targetUpload);
  if (el.voiceWorkSourceUploadBlock) el.voiceWorkSourceUploadBlock.classList.toggle("is-hidden", !sourceUpload);
  if (el.voiceWorkSourceAssetBlock) el.voiceWorkSourceAssetBlock.classList.toggle("is-hidden", sourceUpload);
  state.voiceWorkTargetInputMode = targetUpload ? "upload" : "asset";
  state.voiceWorkSourceInputMode = sourceUpload ? "upload" : "asset";
  applyVoiceWorkAvailability();
}

function selectedVoiceWorkAsset(select) {
  const assetId = select.value;
  return voiceWorkAssetEntries().find((asset) => asset.asset_id === assetId) || null;
}

function populateVoiceWorkVoiceSelectors() {
  const voices = Array.isArray(state.voiceVoices) ? state.voiceVoices : [];
  const selected = state.selectedVoiceWorkVoiceId || "";
  const options = voices.map((voice) => ({
    value: voice.voice_id,
    label: `${voice.label || voice.voice_id}${voice.language ? ` (${voice.language})` : ""}`,
  }));
  for (const select of [el.voiceWorkSampleVoiceSelect]) {
    if (!select) continue;
    const current = select.value;
    select.replaceChildren();
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Choose a target voice";
    select.appendChild(placeholder);
    options.forEach((optionData) => {
      const option = document.createElement("option");
      option.value = optionData.value;
      option.textContent = optionData.label;
      select.appendChild(option);
    });
    if (current && options.some((option) => option.value === current)) {
      select.value = current;
    } else if (selected && options.some((option) => option.value === selected)) {
      select.value = selected;
    } else if (options.length) {
      select.value = options[0].value;
    } else {
      select.value = "";
    }
  }
}

function renderVoiceWorkTrainingRecords() {
  if (!el.voiceWorkTrainingList) return;
  el.voiceWorkTrainingList.replaceChildren();
}

function renderVoiceWorkGenerations() {
  el.voiceWorkGenerationList.replaceChildren();
  if (!state.voiceGenerations.length) {
    const empty = document.createElement("div");
    empty.className = "empty-result";
    empty.textContent = "No voice outputs yet.";
    el.voiceWorkGenerationList.appendChild(empty);
    return;
  }
  state.voiceGenerations.forEach((generation) => {
    const row = document.createElement("article");
    row.className = "generated-item";
    const audioPath = generation.output_audio_path || "";
    const audio = audioPath
      ? `<audio controls preload="metadata" src="/api/audio?path=${encodeURIComponent(audioPath)}"></audio>`
      : `<div class="empty-result">No audio saved.</div>`;
    row.innerHTML = `
      <div class="generated-title">
        <strong>${escapeHtml(generation.render_type || "voice")}</strong>
        <span>${escapeHtml(generation.label || generation.generation_id || "voice output")}</span>
      </div>
      ${audio}
      <div class="summary">${escapeHtml(generation.voice_label || generation.voice_id || "No voice")}${generation.text ? `<br>${escapeHtml(generation.text)}` : ""}</div>
    `;
    el.voiceWorkGenerationList.appendChild(row);
  });
}

function renderVoiceWorkVoices() {
  el.voiceWorkList.replaceChildren();
  const voices = Array.isArray(state.voiceVoices) ? state.voiceVoices : [];
  setPill(el.voiceWorkState, voices.length ? `${voices.length} voices` : "No voices", voices.length ? "ok" : "neutral");
  if (!voices.length) {
    const empty = document.createElement("div");
    empty.className = "empty-result";
    empty.textContent = "No target voices yet.";
    el.voiceWorkList.appendChild(empty);
    renderVoiceWorkSelection();
    populateVoiceWorkVoiceSelectors();
    return;
  }
  voices.forEach((voice, index) => {
    const row = document.createElement("article");
    row.className = "generated-item";
    const previewPath = voice.preview_audio_path || "";
    const audio = previewPath
      ? `<audio controls preload="metadata" src="/api/audio?path=${encodeURIComponent(previewPath)}"></audio>`
      : `<div class="empty-result">No preview audio stored.</div>`;
    row.innerHTML = `
      <div class="generated-title">
        <strong>${index === 0 ? "Latest" : "Voice"} - ${escapeHtml(voice.language || "auto")}</strong>
        <span>${escapeHtml(voice.label || voice.voice_id || "voice")}</span>
      </div>
      ${audio}
      <div class="summary">${escapeHtml(voice.description || "No description")}<br>Status: ${escapeHtml(String(voice.training_status || "ready"))}${voice.source_asset_label ? `<br>Source: ${escapeHtml(voice.source_asset_label)}${voice.source_asset_category ? ` (${escapeHtml(voice.source_asset_category)})` : ""}` : ""}</div>
      <div class="button-row generated-actions">
        <button class="secondary-button voice-work-select-button" type="button">${voice.voice_id === state.selectedVoiceWorkVoiceId ? "Selected" : "Use Voice"}</button>
        <button class="secondary-button voice-work-delete-button" type="button">Delete</button>
      </div>
    `;
    const selectButton = row.querySelector(".voice-work-select-button");
    const deleteButton = row.querySelector(".voice-work-delete-button");
    const busy = voiceWorkBusy();
    selectButton.disabled = busy || voice.voice_id === state.selectedVoiceWorkVoiceId;
    deleteButton.disabled = busy;
    selectButton.addEventListener("click", () => {
      state.selectedVoiceWorkVoiceId = voice.voice_id;
      populateVoiceWorkVoiceInputs(voice);
      renderVoiceWorkVoices();
      renderVoiceWorkSelection();
      applyVoiceWorkAvailability();
    });
    deleteButton.addEventListener("click", () => {
      deleteVoiceWorkVoice(voice.voice_id).catch((error) => showToast(error.message));
    });
    el.voiceWorkList.appendChild(row);
  });
  populateVoiceWorkVoiceSelectors();
  renderVoiceWorkSelection();
}

function renderMusicLokrAdapters() {
  if (!el.musicLokrAdapterSelect) return;
  const current = el.musicLokrAdapterSelect.value;
  el.musicLokrAdapterSelect.replaceChildren();
  const none = document.createElement("option");
  none.value = "";
  none.textContent = "No LoKr";
  el.musicLokrAdapterSelect.appendChild(none);
  state.lokrAdapters.forEach((adapter) => {
    const option = document.createElement("option");
    option.value = adapter.adapter_id;
    option.textContent = `${adapter.label || adapter.adapter_id} - ${adapter.model || "model"}`;
    el.musicLokrAdapterSelect.appendChild(option);
  });
  if (current && state.lokrAdapters.some((adapter) => adapter.adapter_id === current)) {
    el.musicLokrAdapterSelect.value = current;
  }
}

function voiceWorkBusy() {
  const runtime = state.voiceWorkStatus || {};
  return Boolean((runtime.action && runtime.action.active) || (runtime.job && runtime.job.active));
}

function voiceWorkRuntimeReady() {
  return Boolean(state.voiceWorkStatus && state.voiceWorkStatus.api_running);
}

function applyVoiceWorkAvailability() {
  const runtimeBusy = Boolean(state.voiceWorkStatus && state.voiceWorkStatus.action && state.voiceWorkStatus.action.active);
  const jobBusy = Boolean(state.voiceWorkStatus && state.voiceWorkStatus.job && state.voiceWorkStatus.job.active);
  const runtimeReady = voiceWorkRuntimeReady();
  const selectedVoice = selectedVoiceWorkVoice();
  const selectedVoiceReady = Boolean(selectedVoice);
  const targetUsingUpload = !(el.voiceWorkTargetAssetMode && el.voiceWorkTargetAssetMode.checked);
  const sourceUsingUpload = !(el.voiceWorkSourceAssetMode && el.voiceWorkSourceAssetMode.checked);
  const targetAssetChosen = Boolean(selectedVoiceWorkAsset(el.voiceWorkAssetSelect));
  const sourceAssetChosen = Boolean(selectedSourceAsset(el.voiceWorkSourceAssetSelect));
  const targetUploadReady = Boolean(el.voiceWorkReferenceFiles && el.voiceWorkReferenceFiles.files && el.voiceWorkReferenceFiles.files.length);
  const sourceUploadReady = Boolean(el.voiceWorkSampleFile && el.voiceWorkSampleFile.files && el.voiceWorkSampleFile.files.length);
  const disableRuntimeButtons = runtimeBusy;
  const disableJobButtons = runtimeBusy || jobBusy || !runtimeReady;
  if (el.installVoiceWorkRuntimeButton) el.installVoiceWorkRuntimeButton.disabled = disableRuntimeButtons;
  if (el.startVoiceWorkRuntimeButton) {
    const runtime = state.voiceWorkStatus || {};
    const runtimeBooting = runtime.phase === "starting" || runtime.phase === "stale";
    el.startVoiceWorkRuntimeButton.disabled = disableRuntimeButtons || runtimeBooting || runtime.phase === "missing";
  }
  if (el.stopVoiceWorkRuntimeButton) {
    const runtime = state.voiceWorkStatus || {};
    el.stopVoiceWorkRuntimeButton.disabled = disableRuntimeButtons || (!runtime.api_running && !runtime.managed_pid_alive);
  }
  if (el.createVoiceWorkCloneButton) el.createVoiceWorkCloneButton.disabled = disableRuntimeButtons || (targetUsingUpload ? !targetUploadReady : !targetAssetChosen);
  if (el.updateVoiceWorkButton) el.updateVoiceWorkButton.disabled = !state.selectedVoiceWorkVoiceId;
  if (el.convertVoiceWorkSampleButton) el.convertVoiceWorkSampleButton.disabled = disableJobButtons || !selectedVoiceReady || (sourceUsingUpload ? !sourceUploadReady : !sourceAssetChosen);
  if (el.voiceWorkImportAssetButton) el.voiceWorkImportAssetButton.disabled = false;
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
  const generations = await api("/api/music-generations");
  state.musicResults = generations.filter((item) => item.type !== "vocal2bgm" && item.type !== "sound_effect");
  renderMusicList();
}

async function refreshVocal2BgmGenerations() {
  state.soundEffectResults = await api("/api/sound-effects");
  renderSoundEffectGenerations();
}

async function refreshVoiceWorkStatus() {
  const wasJobActive = Boolean(state.voiceWorkStatus && state.voiceWorkStatus.job && state.voiceWorkStatus.job.active);
  state.voiceWorkStatus = await api("/api/voice-work/status");
  renderVoiceWorkRuntime();
  const jobActive = Boolean(state.voiceWorkStatus && state.voiceWorkStatus.job && state.voiceWorkStatus.job.active);
  if (jobActive) {
    scheduleVoiceWorkRuntimePolling();
  } else if (wasJobActive || (state.voiceWorkStatus && state.voiceWorkStatus.job && state.voiceWorkStatus.job.completed_at)) {
    await Promise.all([refreshVoiceWorkVoices(), refreshVoiceWorkGenerations()]);
  }
}

function stopVoiceWorkRuntimePolling() {
  if (state.voiceWorkRuntimePollTimer) {
    window.clearTimeout(state.voiceWorkRuntimePollTimer);
    state.voiceWorkRuntimePollTimer = null;
  }
}

function scheduleVoiceWorkRuntimePolling() {
  stopVoiceWorkRuntimePolling();
  const actionActive = Boolean(state.voiceWorkStatus && state.voiceWorkStatus.action && state.voiceWorkStatus.action.active);
  const jobActive = Boolean(state.voiceWorkStatus && state.voiceWorkStatus.job && state.voiceWorkStatus.job.active);
  if (!actionActive && !jobActive) return;
  state.voiceWorkRuntimePollTimer = window.setTimeout(async () => {
    try {
      await refreshVoiceWorkStatus();
      const stillActionActive = Boolean(state.voiceWorkStatus && state.voiceWorkStatus.action && state.voiceWorkStatus.action.active);
      const stillJobActive = Boolean(state.voiceWorkStatus && state.voiceWorkStatus.job && state.voiceWorkStatus.job.active);
      if (stillActionActive || stillJobActive) {
        scheduleVoiceWorkRuntimePolling();
      } else {
        await Promise.all([refreshVoiceWorkVoices(), refreshVoiceWorkGenerations()]);
        stopVoiceWorkRuntimePolling();
      }
    } catch (error) {
      stopVoiceWorkRuntimePolling();
      showToast(error.message);
    }
  }, 2000);
}

async function refreshVoiceWorkVoices() {
  state.voiceVoices = await api("/api/voice-work/voices");
  if (state.selectedVoiceWorkVoiceId && !state.voiceVoices.some((voice) => voice.voice_id === state.selectedVoiceWorkVoiceId)) {
    state.selectedVoiceWorkVoiceId = null;
  }
  if (!state.selectedVoiceWorkVoiceId && state.voiceVoices.length) {
    state.selectedVoiceWorkVoiceId = state.voiceVoices[0].voice_id;
    populateVoiceWorkVoiceInputs(state.voiceVoices[0]);
  }
  if (state.selectedVoiceWorkVoiceId) {
    populateVoiceWorkVoiceInputs(selectedVoiceWorkVoice());
  }
  renderVoiceWorkVoices();
  renderVoiceWorkTrainingRecords();
  renderVoiceWorkAssetOptions();
  applyVoiceWorkAvailability();
}

async function refreshVoiceWorkGenerations() {
  state.voiceGenerations = await api("/api/voice-work/generations");
  renderVoiceWorkGenerations();
}

async function deleteVoiceWorkVoice(voiceId) {
  const voice = state.voiceVoices.find((item) => item.voice_id === voiceId) || null;
  const label = voice ? voice.label || voice.voice_id : voiceId;
  if (!window.confirm(`Delete target voice "${label}"?`)) return;
  await api(`/api/voice-work/voices/${encodeURIComponent(voiceId)}`, { method: "DELETE" });
  if (state.selectedVoiceWorkVoiceId === voiceId) state.selectedVoiceWorkVoiceId = null;
  await Promise.all([refreshVoiceWorkVoices(), refreshVoiceWorkGenerations()]);
  showToast("Target voice deleted");
}

async function updateVoiceWorkVoice() {
  const voiceId = state.selectedVoiceWorkVoiceId;
  if (!voiceId) {
    showToast("Select a target voice first");
    return;
  }
  const payload = new FormData();
  payload.set("label", el.voiceWorkLabel.value.trim());
  payload.set("language", el.voiceWorkLanguage.value.trim() || "auto");
  payload.set("description", el.voiceWorkDescription.value.trim());
  await fetch(`/api/voice-work/voices/${encodeURIComponent(voiceId)}`, {
    method: "PATCH",
    body: payload,
  }).then(async (response) => {
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(formatApiDetail(data.detail || data.error || `Request failed: ${response.status}`));
    }
    state.selectedVoiceWorkVoiceId = data.voice.voice_id;
  });
  await refreshVoiceWorkVoices();
  showToast("Target voice updated");
}

async function trainVoiceWorkVoice() {
  throw new Error("Voice training has been removed. Upload a target voice or use an asset instead.");
}

async function createVoiceWorkTargetVoiceFromFiles() {
  const label = el.voiceWorkLabel ? el.voiceWorkLabel.value.trim() : "";
  const language = el.voiceWorkLanguage ? el.voiceWorkLanguage.value.trim() || "auto" : "auto";
  const description = el.voiceWorkDescription ? el.voiceWorkDescription.value.trim() : "";
  const files = Array.from((el.voiceWorkReferenceFiles && el.voiceWorkReferenceFiles.files) || []);
  const usingUpload = !(el.voiceWorkTargetAssetMode && el.voiceWorkTargetAssetMode.checked);
  const selectedAsset = selectedVoiceWorkAsset(el.voiceWorkAssetSelect);
  console.info("[Voice Work] Add Target Voice clicked", {
    label,
    language,
    descriptionLength: description.length,
    referenceFileCount: files.length,
    usingUpload,
    selectedAsset: selectedAsset ? { assetId: selectedAsset.asset_id, label: selectedAsset.label, category: selectedAsset.category } : null,
    selectedVoiceId: state.selectedVoiceWorkVoiceId || null,
  });
  if (!label) {
    console.warn("[Voice Work] Add Target Voice blocked: missing label");
    showToast("Enter a target voice label");
    return;
  }
  if (usingUpload && !files.length) {
    console.warn("[Voice Work] Add Target Voice blocked: missing reference audio");
    showToast("Choose at least one reference audio file");
    return;
  }
  if (!usingUpload && !selectedAsset) {
    console.warn("[Voice Work] Add Target Voice blocked: missing existing creation");
    showToast("Choose an existing creation");
    return;
  }
  setPill(el.voiceWorkState, "Saving", "warn");
  showToast("Saving target voice");
  try {
    if (usingUpload) {
      const form = new FormData();
      form.set("label", label);
      form.set("language", language);
      form.set("description", description);
      files.forEach((file) => form.append("files", file, file.name));
      console.info("[Voice Work] Uploading target voice references", {
        fileNames: files.map((file) => file.name),
      });
      const response = await fetch("/api/voice-work/voices/upload", {
        method: "POST",
        body: form,
      });
      const payload = await response.json().catch(() => ({}));
      console.info("[Voice Work] Add Target Voice response", {
        ok: response.ok,
        status: response.status,
        payload,
      });
      if (!response.ok) {
        throw new Error(formatApiDetail(payload.detail || payload.error || `Request failed: ${response.status}`));
      }
      state.selectedVoiceWorkVoiceId = payload.voice.voice_id;
      populateVoiceWorkVoiceInputs(payload.voice);
      populateVoiceWorkVoiceSelectors();
      renderVoiceWorkSelection();
      if (el.voiceWorkReferenceFiles) el.voiceWorkReferenceFiles.value = "";
      if (el.voiceWorkReferenceFilesName) el.voiceWorkReferenceFilesName.textContent = "No files selected";
      await refreshVoiceWorkVoices();
      await refreshVoiceWorkStatus();
      await refreshLocalLibrary();
      showToast("Target voice saved");
      return;
    }
    const response = await api("/api/voice-work/voices/from-asset", {
      method: "POST",
      body: JSON.stringify({
        asset_id: selectedAsset.asset_id,
        label,
        language,
        description,
      }),
    });
    state.selectedVoiceWorkVoiceId = response.voice.voice_id;
    populateVoiceWorkVoiceInputs(response.voice);
    populateVoiceWorkVoiceSelectors();
    renderVoiceWorkSelection();
    await Promise.all([refreshVoiceWorkVoices(), refreshVoiceWorkStatus(), refreshLocalLibrary()]);
    showToast("Target voice imported from asset");
  } catch (error) {
    await refreshVoiceWorkStatus();
    throw error;
  }
}

window.voiceWorkCreateTargetVoiceFromFiles = () => createVoiceWorkTargetVoiceFromFiles();

async function importVoiceWorkTargetFromAsset() {
  const assetId = el.voiceWorkAssetSelect ? el.voiceWorkAssetSelect.value : "";
  if (!assetId) {
    showToast("Choose an existing creation");
    return;
  }
  const asset = selectedVoiceWorkAsset(el.voiceWorkAssetSelect);
  showToast("Saving target voice");
  console.info("[Voice Work] Import target voice from asset clicked", {
    assetId,
    assetLabel: asset && (asset.label || asset.title) || null,
  });
  const response = await api("/api/voice-work/voices/from-asset", {
    method: "POST",
    body: JSON.stringify({
      asset_id: assetId,
      label: el.voiceWorkLabel.value.trim() || (asset && (asset.label || asset.title)) || "",
      language: el.voiceWorkLanguage.value.trim() || "auto",
      description: el.voiceWorkDescription.value.trim(),
    }),
  });
  state.selectedVoiceWorkVoiceId = response.voice.voice_id;
  populateVoiceWorkVoiceInputs(response.voice);
  await Promise.all([refreshVoiceWorkVoices(), refreshLocalLibrary()]);
  showToast("Target voice imported from asset");
}

window.voiceWorkImportTargetFromAsset = () => importVoiceWorkTargetFromAsset();

async function convertVoiceWorkSample() {
  const voiceId = el.voiceWorkSampleVoiceSelect ? el.voiceWorkSampleVoiceSelect.value : state.selectedVoiceWorkVoiceId;
  if (!voiceId) {
    showToast("Select a target voice first");
    return;
  }
  const label = el.voiceWorkSampleLabel.value.trim();
  if (!label) {
    showToast("Enter an output label");
    return;
  }
  const mode = el.voiceWorkSampleMode ? el.voiceWorkSampleMode.value || "singing" : "singing";
  const usingUpload = !(el.voiceWorkSourceAssetMode && el.voiceWorkSourceAssetMode.checked);
  const selectedAsset = selectedSourceAsset(el.voiceWorkSourceAssetSelect);
  const file = el.voiceWorkSampleFile.files && el.voiceWorkSampleFile.files[0];
  if (usingUpload && !file) {
    showToast("Choose a source audio file");
    return;
  }
  if (!usingUpload && !selectedAsset) {
    showToast("Choose an existing creation");
    return;
  }
  const requestId =
    typeof crypto !== "undefined" && crypto.randomUUID
      ? crypto.randomUUID().replace(/-/g, "").slice(0, 12)
      : `vw-${Math.random().toString(16).slice(2, 14)}`;
  state.voiceWorkStatus = {
    ...(state.voiceWorkStatus || {}),
    job: {
      active: true,
      action: "convert",
      message: `Converting in progress. Request ${requestId}.`,
      details: { request_id: requestId, voice_id: voiceId },
    },
  };
  applyVoiceWorkAvailability();
  renderVoiceWorkRuntime();
  scheduleVoiceWorkRuntimePolling();
  try {
    let sourceAudioPath = "";
    let sourceLabel = "";
    if (!usingUpload && selectedAsset) {
      sourceAudioPath = selectedAsset.audio_path;
      sourceLabel = `${selectedAsset.category || "asset"}: ${selectedAsset.label || selectedAsset.asset_id || "source"}`;
      console.info("[Voice Work] Using source asset for conversion", {
        assetId: selectedAsset.asset_id || selectedAsset.id || "",
        assetLabel: selectedAsset.label || selectedAsset.title || "",
        audioPath: sourceAudioPath,
      });
    } else {
      const form = new FormData();
      form.set("file", file, file.name);
      const uploaded = await fetch("/api/voice-work/tmp-upload", {
        method: "POST",
        body: form,
      }).catch(() => null);
      if (!uploaded || !uploaded.ok) {
        throw new Error("Could not stage source audio");
      }
      const payload = await uploaded.json();
      sourceAudioPath = payload.path;
      sourceLabel = payload.name || file.name;
    }
    const request = {
      request_id: requestId,
      voice_id: voiceId,
      source_audio_path: sourceAudioPath,
      label,
      mode,
      diffusion_steps: Number(el.voiceWorkSampleDiffusionSteps.value || 25),
      length_adjust: Number(el.voiceWorkSampleLengthAdjust.value || 1),
      inference_cfg_rate: Number(el.voiceWorkSampleCfgRate.value || 0.7),
    };
    const result = await api(`/api/voice-work/voices/${encodeURIComponent(voiceId)}/convert`, {
      method: "POST",
      body: JSON.stringify(request),
    });
    await refreshVoiceWorkStatus();
    await refreshVoiceWorkGenerations();
    console.info("[Voice Work] Conversion mode used", { mode });
    if (el.voiceWorkSampleFileName && selectedAsset) {
      el.voiceWorkSampleFileName.textContent = sourceLabel;
    }
    showToast(`Voice conversion started (${requestId})`);
    return result;
  } catch (error) {
    await refreshVoiceWorkStatus();
    throw error;
  }
}

async function triggerVoiceWorkRuntimeAction(action) {
  const endpoint =
    action === "install"
      ? "/api/voice-work/runtime/install"
      : action === "restart"
        ? "/api/voice-work/runtime/restart"
        : "/api/voice-work/runtime/start";
  await api(endpoint, { method: "POST" });
  await refreshVoiceWorkStatus();
  scheduleVoiceWorkRuntimePolling();
  showToast(
    action === "install"
      ? "Seed-VC runtime install started"
      : action === "restart"
        ? "Seed-VC runtime restart started"
        : "Seed-VC runtime start started",
    );
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
  const [status, runtime, voiceRuntime, presets, models, tracks, extractions, musicGenerations, soundEffects, voiceGenerations, voiceVoices, lokrDatasets, datasetSources, lokrRuns, lokrAdapters, instrumentClips, rhythmProjects, rhythmVolumes, editorAssets, localLibrary, libraryConnection, logs] = await Promise.all([
    api("/api/status"),
    api("/api/runtime/status"),
    api("/api/voice-work/status"),
    api("/api/presets"),
    api("/api/models"),
    api("/api/extractions/tracks"),
    api("/api/extractions"),
    api("/api/music-generations"),
    api("/api/sound-effects"),
    api("/api/voice-work/generations"),
    api("/api/voice-work/voices"),
    api("/api/lokr/datasets"),
    api("/api/lokr/dataset-sources"),
    api("/api/lokr/runs"),
    api("/api/lokr/adapters"),
    api("/api/instrument-lab/clips"),
    api("/api/rhythm-beats/projects"),
    api("/api/rhythm-beats/volumes"),
    api("/api/editor/assets"),
    api("/api/library/local"),
    api("/api/library/publish/connection"),
    api("/api/logs"),
  ]);
  state.presets = presets;
  state.models = models;
  state.extractionTracks = tracks;
  state.extractionResults = extractions;
  state.musicResults = musicGenerations.filter((item) => item.type !== "vocal2bgm" && item.type !== "sound_effect");
  state.soundEffectResults = soundEffects;
  state.voiceGenerations = voiceGenerations;
  state.voiceWorkStatus = voiceRuntime;
  state.voiceVoices = voiceVoices;
  if (!state.selectedVoiceWorkVoiceId && state.voiceVoices.length) {
    state.selectedVoiceWorkVoiceId = state.voiceVoices[0].voice_id;
    populateVoiceWorkVoiceInputs(state.voiceVoices[0]);
  }
  state.lokrDatasets = lokrDatasets;
  state.datasetSources = datasetSources;
  state.lokrRuns = lokrRuns;
  state.lokrAdapters = lokrAdapters;
  state.instrumentClips = instrumentClips;
  state.rhythmBeatProjects = rhythmProjects;
  state.rhythmBeatVolumes = rhythmVolumes.volumes || [];
  state.editorAssets = editorAssets;
  state.localLibraryItems = localLibrary.items || [];
  state.localLibraryIndexPath = localLibrary.index_path || "";
  state.publicLibraryConnection = libraryConnection;
  renderStatus(status);
  renderRuntime(runtime);
  renderPresets();
  renderModels();
  renderExtractionTracks();
  renderExtractionList();
  applyMusicModelDefaults();
  syncMusicVocalControls();
  renderMusicLokrAdapters();
  renderMusicList();
  renderSoundEffectGenerations();
  renderVoiceWorkRuntime();
  renderVoiceWorkVoices();
  renderVoiceWorkTrainingRecords();
  renderVoiceWorkGenerations();
  scheduleVoiceWorkRuntimePolling();
  renderLokrDatasets();
  renderLokrDatasetEditor();
  renderDatasetEditorSources();
  renderDatasetEditorTarget();
  renderDatasetEditorDonor();
  renderLokrRuns();
  renderInstrumentTracks();
  renderInstrumentPianoKeys();
  drawInstrumentPianoRoll();
  renderInstrumentClipList();
  renderSourceAssetOptions();
  renderEditorAssets();
  renderLocalLibrary();
  renderVoiceWorkAssetOptions();
  renderVoiceWorkInputModes();
  renderPublicLibrary();
  if (runtime.recovery && runtime.recovery.active) {
    startRuntimeRecoveryPolling();
  }
  if (state.rhythmBeatProjects.length) {
    await loadRhythmProject(state.rhythmBeatProjects[0].project_id, false);
  } else {
    renderRhythmBeatLab();
  }
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
    const recoveryActive = Boolean(response.extraction.runtime_recovery && response.extraction.runtime_recovery.active);
    if (response.extraction.status === "complete") {
      if (recoveryActive) {
        setPill(el.extractActionState, "Recovering", "warn");
        el.extractionActivity.innerHTML = "<strong>Recovering</strong><br>Track extraction finished. ACE-Step is restarting to release memory.";
        startRuntimeRecoveryPolling();
      } else {
        setPill(el.extractActionState, "Complete", "ok");
        el.extractionActivity.innerHTML = "<strong>Complete</strong><br>Track extraction finished.";
      }
    } else if (response.extraction.status === "recovering") {
      setPill(el.extractActionState, "Recovering", "warn");
      el.extractionActivity.innerHTML = `<strong>Recovering</strong><br>${escapeHtml(response.extraction.message)}`;
      startRuntimeRecoveryPolling();
    } else {
      setPill(el.extractActionState, "Failed", "error");
      el.extractionActivity.innerHTML = `<strong>Failed</strong><br>${escapeHtml(response.extraction.message)}`;
    }
    showToast(response.extraction.message);
  } catch (error) {
    setPill(el.extractActionState, "Error", "error");
    el.extractionActivity.innerHTML = `<strong>Error</strong><br>${escapeHtml(error.message)}`;
    showToast(error.message);
  } finally {
    await refreshRuntimeState().catch(() => {});
    applyAceRuntimeAvailability();
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
        lokr_adapter_id: el.musicLokrAdapterSelect.value || null,
        lokr_scale: numericValue(el.musicLokrScale) ?? 1,
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
    await refreshLocalLibrary();
    if (response.generation.status === "complete") {
      setPill(el.musicActionState, "Complete", "ok");
      el.musicActivity.innerHTML = "<strong>Complete</strong><br>Music generation finished.";
    } else if (response.generation.status === "recovering") {
      setPill(el.musicActionState, "Recovering", "warn");
      el.musicActivity.innerHTML = `<strong>Recovering</strong><br>${escapeHtml(response.generation.message)}`;
      startRuntimeRecoveryPolling();
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
    await refreshRuntimeState().catch(() => {});
    applyAceRuntimeAvailability();
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
    el.generationActivity.innerHTML = `<strong>Error</strong><br>${escapeHtml(error.message)}`;
    showToast(error.message);
  } finally {
    stopGenerationPolling();
    await refreshRuntimeState().catch(() => {});
    applyAceRuntimeAvailability();
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
  await refreshRuntimeState();
  await refreshActivity();
  await refreshModels();
  await refreshEditorAssets();
  await refreshMusicGenerations();
  await refreshVocal2BgmGenerations();
  await refreshRhythmProjects();
  await refreshLokrRuns();
  await refreshLogs();
  showToast("Status refreshed");
}

el.transitionTabButton.addEventListener("click", () => setActivePage("transition"));
el.extractionTabButton.addEventListener("click", () => setActivePage("extraction"));
el.musicTabButton.addEventListener("click", () => setActivePage("music"));
el.voiceWorkTabButton.addEventListener("click", () => setActivePage("voice"));
el.datasetEditorTabButton.addEventListener("click", () => setActivePage("dataseteditor"));
el.lokrTrainingTabButton.addEventListener("click", () => setActivePage("lokr"));
el.instrumentLabTabButton.addEventListener("click", () => setActivePage("instrument"));
el.audioEditTabButton.addEventListener("click", () => setActivePage("audioedit"));
el.rhythmBeatTabButton.addEventListener("click", () => setActivePage("rhythm"));
el.libraryTabButton.addEventListener("click", () => setActivePage("library"));
el.reloadAudioEditorButton.addEventListener("click", reloadAudioEditor);
el.openAudioEditorButton.addEventListener("click", openAudioEditorWindow);
el.refreshEditorAssetsButton.addEventListener("click", async () => {
  await refreshEditorAssets();
  showToast("Editor assets refreshed");
});
el.refreshLibraryButton.addEventListener("click", async () => {
  try {
    await reindexLocalLibrary();
  } catch (error) {
    showToast(error.message);
  }
});
el.reindexLibraryButton.addEventListener("click", async () => {
  try {
    await reindexLocalLibrary();
  } catch (error) {
    showToast(error.message);
  }
});
el.datasetEditorCreateButton.addEventListener("click", () => {
  createDatasetEditorTarget().catch((error) => showToast(error.message));
});
el.datasetEditorJsonFile.addEventListener("change", () => {
  const file = selectedImportJsonFile(el.datasetEditorJsonFile);
  el.datasetEditorJsonFileName.textContent = file ? file.name : "No JSON selected";
});
el.datasetEditorCreateFromJsonButton.addEventListener("click", () => {
  createDatasetEditorTargetFromJson().catch((error) => showToast(error.message));
});
el.datasetEditorAppendJsonButton.addEventListener("click", () => {
  appendDatasetEditorJson().catch((error) => showToast(error.message));
});
el.datasetEditorRefreshButton.addEventListener("click", () => {
  refreshDatasetSources().then(() => showToast("Dataset sources refreshed")).catch((error) => showToast(error.message));
});
el.datasetEditorSaveButton.addEventListener("click", () => {
  saveDatasetEditorTarget().catch((error) => showToast(error.message));
});
el.connectLibraryWalletButton.addEventListener("click", () => connectLibraryWallet());
el.disconnectLibraryWalletButton.addEventListener("click", () => disconnectLibraryWallet());
el.libraryWalletProvider.addEventListener("change", () => {
  setPreferredLibraryWallet(el.libraryWalletProvider.value || "phantom");
  renderLibraryConnection();
});
el.refreshPublicLibraryButton.addEventListener("click", () => {
  refreshPublicLibrary().catch((error) => showToast(error.message));
});
el.publicLibraryKind.addEventListener("change", () => {
  if (state.publicLibraryItems.length) refreshPublicLibrary().catch((error) => showToast(error.message));
});
el.publicLibrarySearch.addEventListener("input", renderPublicLibrary);
if (el.voiceWorkReferenceFiles) {
  el.voiceWorkReferenceFiles.addEventListener("change", () => {
    const count = (el.voiceWorkReferenceFiles.files || []).length;
    el.voiceWorkReferenceFilesName.textContent = count ? `${count} file${count === 1 ? "" : "s"} selected` : "No files selected";
    if (count && el.voiceWorkTargetUploadMode) {
      el.voiceWorkTargetUploadMode.checked = true;
      renderVoiceWorkInputModes();
    }
  });
}
if (el.voiceWorkSampleFile) {
  el.voiceWorkSampleFile.addEventListener("change", () => {
    const file = el.voiceWorkSampleFile.files && el.voiceWorkSampleFile.files[0];
    el.voiceWorkSampleFileName.textContent = file ? file.name : "No file selected";
    if (file && el.voiceWorkSourceUploadMode) {
      el.voiceWorkSourceUploadMode.checked = true;
      renderVoiceWorkInputModes();
    }
  });
}
if (el.voiceWorkSampleMode) {
  el.voiceWorkSampleMode.addEventListener("change", () => {
    showToast(`Conversion mode set to ${el.voiceWorkSampleMode.value === "speaking" ? "speaking" : "singing"}`);
  });
}
if (el.voiceWorkSampleVoiceSelect) {
  el.voiceWorkSampleVoiceSelect.addEventListener("change", () => {
    state.selectedVoiceWorkVoiceId = el.voiceWorkSampleVoiceSelect.value || null;
    renderVoiceWorkSelection();
    applyVoiceWorkAvailability();
  });
}
if (el.voiceWorkAssetSelect) {
  el.voiceWorkAssetSelect.addEventListener("change", () => {
    if (el.voiceWorkAssetName) {
      const selected = selectedVoiceWorkAsset(el.voiceWorkAssetSelect);
      el.voiceWorkAssetName.textContent = selected
        ? `${selected.category}: ${selected.label}`
        : "No creation selected";
    }
    if (el.voiceWorkTargetAssetMode && el.voiceWorkAssetSelect.value) {
      el.voiceWorkTargetAssetMode.checked = true;
    }
    renderVoiceWorkInputModes();
    applyVoiceWorkAvailability();
  });
}
if (el.voiceWorkSourceAssetSelect) {
  el.voiceWorkSourceAssetSelect.addEventListener("change", () => {
    if (el.voiceWorkSourceAssetName) {
      const selected = selectedSourceAsset(el.voiceWorkSourceAssetSelect);
      el.voiceWorkSourceAssetName.textContent = selected
        ? `${selected.category}: ${selected.label}`
        : "No source asset selected";
    }
    if (el.voiceWorkSampleFile && el.voiceWorkSourceAssetSelect.value) {
      el.voiceWorkSampleFile.value = "";
      if (el.voiceWorkSampleFileName) el.voiceWorkSampleFileName.textContent = "No file selected";
    }
    if (el.voiceWorkSourceAssetMode && el.voiceWorkSourceAssetSelect.value) {
      el.voiceWorkSourceAssetMode.checked = true;
      renderVoiceWorkInputModes();
    }
    applyVoiceWorkAvailability();
  });
}
if (el.voiceWorkTargetUploadMode) {
  el.voiceWorkTargetUploadMode.addEventListener("change", renderVoiceWorkInputModes);
}
if (el.voiceWorkTargetAssetMode) {
  el.voiceWorkTargetAssetMode.addEventListener("change", renderVoiceWorkInputModes);
}
if (el.voiceWorkSourceUploadMode) {
  el.voiceWorkSourceUploadMode.addEventListener("change", renderVoiceWorkInputModes);
}
if (el.voiceWorkSourceAssetMode) {
  el.voiceWorkSourceAssetMode.addEventListener("change", renderVoiceWorkInputModes);
}
el.installVoiceWorkRuntimeButton.addEventListener("click", () => {
  triggerVoiceWorkRuntimeAction("install").catch((error) => showToast(error.message));
});
el.startVoiceWorkRuntimeButton.addEventListener("click", () => {
  const action = state.voiceWorkStatus && state.voiceWorkStatus.simple_start_command === "Restart Runtime" ? "restart" : "start";
  triggerVoiceWorkRuntimeAction(action).catch((error) => showToast(error.message));
});
if (el.stopVoiceWorkRuntimeButton) {
  el.stopVoiceWorkRuntimeButton.addEventListener("click", () => {
    api("/api/voice-work/runtime/stop", { method: "POST" })
      .then(async () => {
        await refreshVoiceWorkStatus();
        stopVoiceWorkRuntimePolling();
        showToast("Seed-VC runtime stopped");
      })
      .catch((error) => showToast(error.message));
  });
}
if (el.updateVoiceWorkButton) {
  el.updateVoiceWorkButton.addEventListener("click", () => {
    updateVoiceWorkVoice().catch((error) => showToast(error.message));
  });
}
if (el.trainVoiceWorkButton) {
  el.trainVoiceWorkButton.addEventListener("click", () => {
    trainVoiceWorkVoice().catch((error) => showToast(error.message));
  });
}
if (el.convertVoiceWorkSampleButton) {
  el.convertVoiceWorkSampleButton.addEventListener("click", () => {
    convertVoiceWorkSample().catch((error) => showToast(error.message));
  });
}
if (el.voiceWorkImportAssetButton) {
  el.voiceWorkImportAssetButton.addEventListener("click", () => {
    importVoiceWorkTargetFromAsset().catch((error) => showToast(error.message));
  });
}
if (el.voiceWorkLoadSourceAssetButton) {
  el.voiceWorkLoadSourceAssetButton.addEventListener("click", () => {
    const selected = selectedSourceAsset(el.voiceWorkSourceAssetSelect);
    if (!selected) {
      showToast("Choose an existing creation");
      return;
    }
    if (el.voiceWorkSourceAssetName) {
      el.voiceWorkSourceAssetName.textContent = `${selected.category}: ${selected.label}`;
    }
    if (el.voiceWorkSampleFile) el.voiceWorkSampleFile.value = "";
    if (el.voiceWorkSampleFileName) el.voiceWorkSampleFileName.textContent = "No file selected";
    showToast("Source asset selected");
  });
}
el.refreshVoiceWorkButton.addEventListener("click", () => {
  Promise.all([refreshVoiceWorkStatus(), refreshVoiceWorkVoices(), refreshVoiceWorkGenerations()])
    .then(() => showToast("Voice Work refreshed"))
    .catch((error) => showToast(error.message));
});
el.editorAssetSearch.addEventListener("input", renderEditorAssets);
el.editorCategoryFilter.addEventListener("change", renderEditorAssets);
el.librarySearch.addEventListener("input", renderLocalLibrary);
el.libraryKindFilter.addEventListener("change", renderLocalLibrary);
el.editSaveFile.addEventListener("change", () => {
  const file = el.editSaveFile.files && el.editSaveFile.files[0];
  el.editSaveFileName.textContent = file ? file.name : "No file selected";
});
el.rhythmSourceFile.addEventListener("change", () => {
  const file = el.rhythmSourceFile.files && el.rhythmSourceFile.files[0];
  el.rhythmSourceFileName.textContent = file ? file.name : "No file selected";
});
el.createRhythmProjectButton.addEventListener("click", () => {
  runWithButtonBusyState(el.createRhythmProjectButton, "Creating...", () => createRhythmProject()).catch((error) => showToast(error.message));
});
el.refreshRhythmProjectsButton.addEventListener("click", () => {
  runWithButtonBusyState(el.refreshRhythmProjectsButton, "Refreshing...", async () => {
    await refreshRhythmProjects();
    showToast("Rhythm beat projects refreshed");
  }).catch((error) => showToast(error.message));
});
el.uploadRhythmSourceButton.addEventListener("click", () => {
  runWithButtonBusyState(el.uploadRhythmSourceButton, "Uploading...", () => uploadRhythmSource()).catch((error) => showToast(error.message));
});
el.loadRhythmSourceAssetButton.addEventListener("click", () => {
  runWithButtonBusyState(el.loadRhythmSourceAssetButton, "Attaching...", () => attachRhythmSourceAsset()).catch((error) => showToast(error.message));
});
el.addRhythmTrackButton.addEventListener("click", () => {
  runWithButtonBusyState(el.addRhythmTrackButton, "Adding...", () => addRhythmTrack()).catch((error) => showToast(error.message));
});
el.runRhythmAnalysisButton.addEventListener("click", () => {
  runWithButtonBusyState(el.runRhythmAnalysisButton, "Analyzing...", () => runRhythmAnalysis()).catch((error) => showToast(error.message));
});
el.runRhythmTrackExtractionButton.addEventListener("click", () => {
  runWithButtonBusyState(el.runRhythmTrackExtractionButton, "Extracting...", () => runRhythmTrackExtraction()).catch((error) => showToast(error.message));
});
el.saveRhythmSelectionButton.addEventListener("click", () => {
  runWithButtonBusyState(el.saveRhythmSelectionButton, "Saving...", () => saveRhythmSelection()).catch((error) => showToast(error.message));
});
el.mergeRhythmSelectionsButton.addEventListener("click", () => {
  runWithButtonBusyState(
    el.mergeRhythmSelectionsButton,
    "Creating...",
    () => mergeRhythmSelections(),
    { restore: () => updateRhythmCandidateActionLabel() }
  ).catch((error) => showToast(error.message));
});
el.finalizeRhythmMergeButton.addEventListener("click", () => {
  runWithButtonBusyState(el.finalizeRhythmMergeButton, "Finalizing...", () => finalizeRhythmMerge()).catch((error) => showToast(error.message));
});
el.saveRhythmLyricsButton.addEventListener("click", () => {
  runWithButtonBusyState(el.saveRhythmLyricsButton, "Saving...", () => saveRhythmLyrics()).catch((error) => showToast(error.message));
});
el.extractRhythmLyricsButton.addEventListener("click", () => {
  runWithButtonBusyState(el.extractRhythmLyricsButton, "Extracting...", () => extractRhythmLyrics()).catch((error) => showToast(error.message));
});
if (el.createRhythmVolumeButton) {
  el.createRhythmVolumeButton.addEventListener("click", () => {
    runWithButtonBusyState(el.createRhythmVolumeButton, "Creating...", () => createRhythmVolume()).catch((error) => showToast(error.message));
  });
}
el.saveRhythmProjectButton.addEventListener("click", () => {
  runWithButtonBusyState(el.saveRhythmProjectButton, "Saving...", () => saveRhythmProject(true)).catch((error) => showToast(error.message));
});
el.rhythmActiveAnalysisSelect.addEventListener("change", () => {
  state.selectedRhythmAnalysisId = el.rhythmActiveAnalysisSelect.value || null;
  syncRhythmPlaybackSourceToAnalysis();
  drawRhythmChart();
});
el.rhythmViewMode.addEventListener("change", drawRhythmChart);
el.rhythmExtractTrackName.addEventListener("change", updateRhythmExtractionLabelPlaceholder);
el.rhythmSourceAudio.addEventListener("timeupdate", drawRhythmChart);
el.rhythmSourceAudio.addEventListener("seeked", drawRhythmChart);
el.rhythmSourceAudio.addEventListener("play", drawRhythmChart);
el.rhythmSourceAudio.addEventListener("pause", drawRhythmChart);
el.rhythmChartStack.addEventListener("pointerdown", (event) => {
  const project = activeRhythmProject();
  if (!project) return;
  const savedSegment = event.target.closest(".rhythm-saved-segment");
  if (savedSegment) {
    event.stopPropagation();
    selectRhythmSavedSegment(
      savedSegment.dataset.selectionId || "",
      Number(savedSegment.dataset.segmentIndex || 0),
    );
    return;
  }
  const draftSegment = event.target.closest(".rhythm-draft-segment");
  if (draftSegment) {
    event.stopPropagation();
    selectRhythmDraftSegment(
      draftSegment.dataset.analysisId || "",
      Number(draftSegment.dataset.segmentIndex || 0),
    );
    return;
  }
  const chartSvg = event.target.closest(".rhythm-chart-svg");
  if (!chartSvg) return;
  const analysisId = chartSvg.dataset.analysisId || "";
  if (!analysisId) return;
  const analysis = (project.analyses || []).find((item) => item.analysis_id === analysisId);
  if (!analysis) return;
  state.selectedRhythmAnalysisId = analysis.analysis_id;
  const time = rhythmClientXToTime(event.clientX);
  const existingDraft = state.rhythmSelectionDrafts[analysis.analysis_id]
    ? { ...state.rhythmSelectionDrafts[analysis.analysis_id], ranges: [...(state.rhythmSelectionDrafts[analysis.analysis_id].ranges || [])] }
    : { analysisId: analysis.analysis_id, ranges: [] };
  state.rhythmSelectionPointer = {
    pointerId: event.pointerId,
    analysisId: analysis.analysis_id,
    startSeconds: time,
    lastSeconds: time,
    moved: false,
  };
  state.rhythmSelectionDrafts[analysis.analysis_id] = existingDraft;
  if (chartSvg.setPointerCapture) {
    chartSvg.setPointerCapture(event.pointerId);
  }
  renderRhythmBeatLab();
});
window.addEventListener("pointermove", (event) => {
  const pointer = state.rhythmSelectionPointer;
  if (!pointer || pointer.pointerId !== event.pointerId) return;
  maybeAutoScrollRhythmChart(event.clientX);
  const time = rhythmClientXToTime(event.clientX);
  pointer.lastSeconds = time;
  if (!pointer.moved && Math.abs(time - pointer.startSeconds) >= 0.03) {
    pointer.moved = true;
    state.rhythmSelectionDrafts[pointer.analysisId] = {
      analysisId: pointer.analysisId,
      ranges: [
        ...(state.rhythmSelectionDrafts[pointer.analysisId]?.ranges || []),
        { startSeconds: pointer.startSeconds, endSeconds: time },
      ],
    };
    state.selectedRhythmDraftSegmentIndices[pointer.analysisId] = state.rhythmSelectionDrafts[pointer.analysisId].ranges.length - 1;
  }
  if (pointer.moved && state.rhythmSelectionDrafts[pointer.analysisId]?.ranges?.length) {
    const draft = state.rhythmSelectionDrafts[pointer.analysisId];
    draft.ranges[draft.ranges.length - 1].endSeconds = time;
  }
  drawRhythmChart();
});
window.addEventListener("pointerup", (event) => {
  const pointer = state.rhythmSelectionPointer;
  if (!pointer || pointer.pointerId !== event.pointerId) return;
  if (!pointer.moved) {
    const nextTime = Math.max(0, pointer.lastSeconds);
    el.rhythmSourceAudio.currentTime = nextTime;
  } else if (state.rhythmSelectionDrafts[pointer.analysisId]?.ranges?.length) {
    const draft = state.rhythmSelectionDrafts[pointer.analysisId];
    const lastIndex = draft.ranges.length - 1;
    const lastRange = draft.ranges[lastIndex];
    draft.ranges[lastIndex] = {
      startSeconds: Math.min(lastRange.startSeconds, lastRange.endSeconds),
      endSeconds: Math.max(lastRange.startSeconds, lastRange.endSeconds),
    };
  }
  state.rhythmSelectionPointer = null;
  drawRhythmChart();
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
if (el.runSoundEffectButton) {
  el.runSoundEffectButton.addEventListener("click", () => {
    runSoundEffectGeneration().catch((error) => showToast(error.message));
  });
}
if (el.refreshSoundEffectButton) {
  el.refreshSoundEffectButton.addEventListener("click", () => {
    refreshSoundEffectGenerations().catch((error) => showToast(error.message));
  });
}
if (el.vocal2bgmSourceUploadMode) {
  el.vocal2bgmSourceUploadMode.addEventListener("change", renderVocal2BgmInputModes);
}
if (el.vocal2bgmSourceAssetMode) {
  el.vocal2bgmSourceAssetMode.addEventListener("change", renderVocal2BgmInputModes);
}
if (el.vocal2bgmPromptInput) {
  el.vocal2bgmPromptInput.addEventListener("input", () => {
    state.vocal2bgmPrompt = el.vocal2bgmPromptInput.value.trim();
  });
}
if (el.vocal2bgmSourceFile) {
  el.vocal2bgmSourceFile.addEventListener("change", () => {
    const file = el.vocal2bgmSourceFile.files && el.vocal2bgmSourceFile.files[0];
    el.vocal2bgmSourceFileName.textContent = file ? file.name : "No file selected";
    if (file) {
      uploadVocal2BgmSourceFile().catch((error) => showToast(error.message));
    }
  });
}
if (el.vocal2bgmSourceAssetSelect) {
  el.vocal2bgmSourceAssetSelect.addEventListener("change", async () => {
    const selected = selectedSourceAsset(el.vocal2bgmSourceAssetSelect);
    if (el.vocal2bgmSourceAssetName) {
      el.vocal2bgmSourceAssetName.textContent = selected
        ? `${selected.category}: ${selected.label}`
        : "No creation selected";
    }
    if (!selected) {
      state.vocal2bgmSourcePath = "";
      state.vocal2bgmSourceProbe = null;
      if (el.vocal2bgmSourceReadout) el.vocal2bgmSourceReadout.textContent = vocal2bgmSourceReadout();
      return;
    }
    if (selected && el.vocal2bgmSourceFile) {
      el.vocal2bgmSourceFile.value = "";
      if (el.vocal2bgmSourceFileName) el.vocal2bgmSourceFileName.textContent = "No file selected";
    }
    if (selected && selected.audio_path) {
      try {
        if (el.vocal2bgmSourceUploadMode) el.vocal2bgmSourceUploadMode.checked = false;
        if (el.vocal2bgmSourceAssetMode) el.vocal2bgmSourceAssetMode.checked = true;
        renderVocal2BgmInputModes();
        await probeVocal2BgmSource(selected.audio_path);
        setPill(el.vocal2bgmActionState, "Ready", "ok");
      } catch (error) {
        setPill(el.vocal2bgmActionState, "Error", "error");
        showToast(error.message);
      }
    }
  });
}
if (el.runVocal2BgmButton) {
  el.runVocal2BgmButton.addEventListener("click", () => {
    runVocal2BgmGeneration().catch((error) => showToast(error.message));
  });
}
if (el.refreshVocal2BgmButton) {
  el.refreshVocal2BgmButton.addEventListener("click", () => {
    refreshVocal2BgmGenerations().catch((error) => showToast(error.message));
  });
}
el.createLokrDatasetButton.addEventListener("click", () => {
  createLokrDataset().catch((error) => showToast(error.message));
});
el.refreshLokrDatasetsButton.addEventListener("click", () => {
  refreshLokrDatasets()
    .then(() => showToast("LoKr datasets refreshed"))
    .catch((error) => showToast(error.message));
});
el.lokrDatasetJsonFile.addEventListener("change", () => {
  const file = selectedImportJsonFile(el.lokrDatasetJsonFile);
  el.lokrDatasetJsonFileName.textContent = file ? file.name : "No JSON selected";
});
el.createLokrDatasetFromJsonButton.addEventListener("click", () => {
  createLokrDatasetFromJson().catch((error) => showToast(error.message));
});
el.appendLokrDatasetJsonButton.addEventListener("click", () => {
  appendLokrDatasetJson().catch((error) => showToast(error.message));
});
el.saveLokrDatasetButton.addEventListener("click", () => {
  saveLokrDataset().catch((error) => showToast(error.message));
});
el.lokrAudioFiles.addEventListener("change", () => {
  const files = el.lokrAudioFiles.files || [];
  el.lokrSelectedFiles.textContent = files.length ? `${files.length} file${files.length === 1 ? "" : "s"} selected` : "No files selected";
  uploadLokrFiles(files).catch((error) => showToast(error.message));
});
el.addLokrAssetButton.addEventListener("click", () => {
  addLokrAsset().catch((error) => showToast(error.message));
});
el.addEmptyLokrEntryButton.addEventListener("click", () => {
  addEmptyLokrEntry().catch((error) => showToast(error.message));
});
el.preprocessLokrButton.addEventListener("click", () => {
  preprocessLokrDataset().catch((error) => showToast(error.message));
});
el.trainLokrButton.addEventListener("click", () => {
  trainLokrDataset().catch((error) => showToast(error.message));
});
el.stopLokrRunButton.addEventListener("click", () => {
  stopLokrRun().catch((error) => showToast(error.message));
});
el.clearLokrLogButton.addEventListener("click", () => {
  clearLokrLogView().catch((error) => showToast(error.message));
});
el.lokrDropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  el.lokrDropZone.classList.add("active");
});
el.lokrDropZone.addEventListener("dragleave", () => {
  el.lokrDropZone.classList.remove("active");
});
el.lokrDropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  el.lokrDropZone.classList.remove("active");
  uploadLokrFiles(event.dataTransfer.files).catch((error) => showToast(error.message));
});
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
window.setInterval(() => {
  if (!state.lokrRuns.some((run) => run.status === "running")) return;
  refreshLokrRuns()
    .then(refreshSelectedLokrRunLog)
    .catch((error) => console.warn("[Dance Station] LoKr run refresh failed", error));
}, 3000);
