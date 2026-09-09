from pathlib import Path


CONFIGURER = Path("/opt/sglang/python/sglang/srt/layers/deep_gemm_wrapper/configurer.py")


def main() -> None:
    source = CONFIGURER.read_text()
    old = """def _compute_enable_deep_gemm():
    sm_version = get_device_sm()
    if sm_version < 90:
        return False

    try:
        import deep_gemm  # noqa: F401
    except ImportError:
        return False

    return envs.SGLANG_ENABLE_JIT_DEEPGEMM.get()
"""
    new = """def _compute_enable_deep_gemm():
    # Do not import DeepGEMM unless it is explicitly enabled. The package
    # initializes its CUDA toolchain during import, which is not needed for
    # MOSS-Music's dense BF16 model and fails in CUDA runtime-only images.
    if not envs.SGLANG_ENABLE_JIT_DEEPGEMM.get():
        return False

    sm_version = get_device_sm()
    if sm_version < 90:
        return False

    try:
        import deep_gemm  # noqa: F401
    except ImportError:
        return False

    return True
"""
    if old not in source:
        raise RuntimeError(f"Unexpected SGLang configurer shape: {CONFIGURER}")
    CONFIGURER.write_text(source.replace(old, new, 1))


if __name__ == "__main__":
    main()
