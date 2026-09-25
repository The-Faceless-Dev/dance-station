from __future__ import annotations

import argparse
import shutil
import time
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download


REPOSITORIES = (
    ("HeartMuLa/MuLaCover", "bbbaef2b31835c2ef5172ff528dbe325230d46fb", "MuLaCover"),
    ("HeartMuLa/HeartCodec-oss-20260123", "f889dab0532cfa4bf459f2a3367eb6d346b8eeda", "HeartCodec-oss"),
    ("Qwen/Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3", "Qwen3-Embedding-0.6B"),
)
# YourMT3 is published from a Hugging Face Space rather than a model repo.  The
# upstream MuLaCover instructions use the Space's main revision; the old commit
# URL returned 404 even though the checkpoint is still present at this path.
YOURMT3_URL = "https://huggingface.co/spaces/mimbres/YourMT3/resolve/main/amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/checkpoints/last.ckpt"
YOURMT3_BYTES = 561_544_628
CHORD_COMMIT = "481f4ce703f8822b99f4037e9104ba1760e21ea3"
CHORD_BYTES = (5_746_183, 5_746_175, 5_746_179, 5_746_175, 5_746_227)


def fetch(url: str, destination: Path, expected: int) -> None:
    if destination.is_file() and destination.stat().st_size == expected:
        print(f"reuse {destination}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, 4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "faceless-dancer-mulacover-worker/1"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream, length=1024 * 1024)
            actual = temporary.stat().st_size
            if actual != expected:
                raise RuntimeError(f"size mismatch: {actual} != {expected}")
            temporary.replace(destination)
            return
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            if attempt == 3:
                raise RuntimeError(f"failed to download {url} after {attempt} attempts: {exc}") from exc
            print(f"retry {attempt}/3 for {destination.name}: {exc}", flush=True)
            time.sleep(attempt * 5)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the pinned MuLaCover model bundle")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    for repo, revision, directory in REPOSITORIES:
        print(f"download {repo}@{revision}", flush=True)
        snapshot_download(repo_id=repo, revision=revision, local_dir=root / directory, max_workers=2)
    fetch(YOURMT3_URL, root / "SymbolicTranscriptor" / "yourmt3" / "last.ckpt", YOURMT3_BYTES)
    for fold, expected in enumerate(CHORD_BYTES):
        filename = f"joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s{fold}.best.sdict"
        fetch(f"https://raw.githubusercontent.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition/{CHORD_COMMIT}/cache_data/{filename}", root / "SymbolicTranscriptor" / "chord" / filename, expected)
    print(f"MuLaCover bundle ready at {root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
