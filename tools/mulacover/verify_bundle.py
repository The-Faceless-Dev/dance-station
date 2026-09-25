from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    required = [
        "MuLaCover/config.json", "MuLaCover/gen_config.json", "MuLaCover/model.safetensors.index.json",
        "MuLaCover/tokenizer.json", "HeartCodec-oss/config.json", "HeartCodec-oss/model.safetensors.index.json",
        "Qwen3-Embedding-0.6B/config.json", "Qwen3-Embedding-0.6B/model.safetensors",
        "SymbolicTranscriptor/yourmt3/last.ckpt",
    ]
    missing = [item for item in required if not (root / item).is_file() or (root / item).stat().st_size == 0]
    for index in ("MuLaCover/model.safetensors.index.json", "HeartCodec-oss/model.safetensors.index.json"):
        if (root / index).is_file():
            data = json.loads((root / index).read_text(encoding="utf-8"))
            for shard in sorted(set(data.get("weight_map", {}).values())):
                path = (root / Path(index).parent / shard).resolve()
                if not path.is_file() or path.stat().st_size == 0:
                    missing.append(str(path.relative_to(root)))
    chord = list((root / "SymbolicTranscriptor/chord").glob("*.best.sdict"))
    if len(chord) != 5:
        missing.append(f"SymbolicTranscriptor/chord/*.best.sdict (found {len(chord)}, expected 5)")
    if missing:
        raise SystemExit("incomplete MuLaCover bundle:\n- " + "\n- ".join(missing))
    print(f"verified MuLaCover bundle: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
