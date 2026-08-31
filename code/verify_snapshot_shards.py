"""Hash completed local Hugging Face snapshot weight shards for provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--revisions", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model_dir = args.cache_dir / ("models--" + args.repo.replace("/", "--"))
    result = {"repo": args.repo, "revisions": {}, "status": "PASS"}
    for revision in args.revisions:
        ref = model_dir / "refs" / revision
        if not ref.is_file():
            result["status"] = "FAIL"
            result["revisions"][revision] = {"status": "MISSING_REF"}
            continue
        snapshot = model_dir / "snapshots" / ref.read_text().strip()
        shards = sorted(snapshot.glob("*.bin")) + sorted(snapshot.glob("*.safetensors"))
        if not shards:
            result["status"] = "FAIL"
            result["revisions"][revision] = {"status": "MISSING_SHARDS"}
            continue
        result["revisions"][revision] = {
            "status": "PASS",
            "snapshot": str(snapshot),
            "files": [
                {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in shards
            ],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
