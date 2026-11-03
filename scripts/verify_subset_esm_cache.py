"""Fingerprint a cache export or verify its byte-identical copy and tensors."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path


def fingerprint(root, names):
    def inspect(name):
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"Path escapes cache: {name}")
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        return name, {"bytes": path.stat().st_size, "sha256": digest}

    with ThreadPoolExecutor(max_workers=4) as workers:
        return dict(workers.map(inspect, names))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--file-list", type=Path)
    parser.add_argument("--expected-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-tensors", action="store_true")
    args = parser.parse_args()
    if bool(args.file_list) == bool(args.expected_manifest):
        parser.error("Provide exactly one of --file-list and --expected-manifest")
    expected = json.loads(args.expected_manifest.read_text()) if args.expected_manifest else None
    names = list(expected) if expected is not None else args.file_list.read_text().splitlines()
    actual = fingerprint(args.cache_dir, names)
    if expected is not None and actual != expected:
        different = [name for name in names if actual[name] != expected[name]]
        raise ValueError(f"Cache transfer changed files: {different[:10]}")
    if args.verify_tensors:
        import torch
        from protenix_embedding_profile import chain_key, load_chain

        torch.set_num_threads(1)
        spec = json.loads((args.cache_dir / "run_spec.json").read_text())
        tensors = [name for name in names if name.endswith(".pt")]
        for index, name in enumerate(tensors, 1):
            key = Path(name).stem
            metadata = json.loads((args.cache_dir / "metadata" / f"{key}.json").read_text())
            if chain_key(metadata["sequence"]) != key:
                raise ValueError(f"Sequence hash mismatch: {key}")
            load_chain(args.cache_dir, metadata["sequence"], spec)
            if index % 1000 == 0 or index == len(tensors):
                print(json.dumps({"verified_tensors": index, "total": len(tensors)}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(actual, indent=2) + "\n")
    print(json.dumps({"files": len(actual), "bytes": sum(value["bytes"] for value in actual.values()),
                      "matches_expected": expected is not None, "verified_tensors": args.verify_tensors}), flush=True)


if __name__ == "__main__":
    main()
