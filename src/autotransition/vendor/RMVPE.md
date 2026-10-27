# RMVPE attribution

`rmvpe.py` is vendored from the MIT-licensed RVC-Project implementation:

<https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/infer/rmvpe.py>

The only runtime adaptation is an optional fallback for the RVC application's
CUDA-graph helper. The YingMusic runtime uses the upstream RMVPE network and
decoder without changing its pitch calculation.
