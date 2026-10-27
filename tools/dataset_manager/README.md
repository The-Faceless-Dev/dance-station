# Vevo2 Dataset Manager

This is the standalone dataset creation app for melodic-vocal Vevo2 adapter
training data. It stores each song, extraction run, vocal stem, alignment
metadata, editable annotation draft, and approved revision under the configured
dataset root.

## Start

Install the project with the dataset extras, then run:

```powershell
pip install -e ".[dataset-manager]"
dataset-manager
```

The default URL is `http://127.0.0.1:7877`. The equivalent development runner
is:

```powershell
python tools/dataset_manager/run.py
```

## Configuration

The app reads `DATASET_MANAGER_*` environment variables. The most important
ones are:

- `DATASET_MANAGER_ROOT`: durable dataset storage directory.
- `DATASET_MANAGER_MODEL_ROOT`: model/cache directory.
- `DATASET_MANAGER_SEPARATOR_RUNTIME`: isolated GPU source-separation runtime.
- `DATASET_MANAGER_SEPARATOR_PRIMARY_MODEL`: primary vocal-separation model.
- `DATASET_MANAGER_NFA_SCRIPT` and `DATASET_MANAGER_NFA_PYTHON`: NeMo Forced
  Aligner entry point and Python environment.
- `DATASET_MANAGER_RMVPE_CHECKPOINT`: optional RMVPE checkpoint for the F0
  feature artifact.

The extraction service requires CUDA for source separation by default and
records the selected provider, device, model, alignment outputs, and warnings
in each run's metadata. The annotation editor works on the extracted vocal
stem and stores exact audio-relative word and phoneme intervals.
