#!/usr/bin/env python3
"""Start epoch-1 negative shards early and finish them after permanent caching."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = Path("/data/luyq/hitph_cycle4_worker")
REMOTE_INPUT = BASE / "job/input/train_negatives_epoch_0001.json"
REMOTE_WORKER = BASE / "epoch1_worker.sh"
INITIAL = (("Tang3", 3), ("Tang1", 0), ("Tang1", 4),
           ("Tang1", 5), ("Tang1", 7), ("LDS_Tang2", 3), ("Tang1", 6))
HOSTS = ("Tang3", "Tang1", "LDS_Tang2")
NUM_SHARDS = 32


def host_settings(config: dict, host: str) -> tuple[str, str]:
    folder = Path(config["output_root"]).name
    if host == "Tang3":
        return config["python"], str(Path(config["output_root"]) / "features")
    return str(BASE / "env/bin/python"), str(BASE / "foldtmp" / folder / "features")


def run_host(host: str, command: str) -> str:
    argv = ["bash", "-lc", command] if host == "Tang3" else ["ssh", "-o", "BatchMode=yes", host, command]
    return subprocess.check_output(argv, text=True).strip()


def stage(config: dict, input_path: Path) -> None:
    worker = Path(__file__).with_name("epoch1_worker.sh")
    expected = hashlib.sha256(input_path.read_bytes()).hexdigest()
    for host in HOSTS:
        if host == "Tang3":
            REMOTE_INPUT.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_path, REMOTE_INPUT.with_suffix(".json.tmp"))
            REMOTE_INPUT.with_suffix(".json.tmp").replace(REMOTE_INPUT)
            shutil.copy2(worker, REMOTE_WORKER.with_suffix(".sh.tmp"))
            REMOTE_WORKER.with_suffix(".sh.tmp").replace(REMOTE_WORKER)
        else:
            subprocess.run(["scp", "-q", str(input_path),
                            f"{host}:{REMOTE_INPUT}.tmp"], check=True)
            subprocess.run(["scp", "-q", str(worker),
                            f"{host}:{REMOTE_WORKER}.tmp"], check=True)
            run_host(host, f"mv {shlex.quote(str(REMOTE_INPUT)+'.tmp')} {shlex.quote(str(REMOTE_INPUT))} && "
                           f"mv {shlex.quote(str(REMOTE_WORKER)+'.tmp')} {shlex.quote(str(REMOTE_WORKER))}")
        actual = run_host(host, f"sha256sum {shlex.quote(str(REMOTE_INPUT))} | cut -d ' ' -f 1")
        if actual != expected:
            raise RuntimeError(f"Epoch-1 input checksum mismatch on {host}")
        print(f"Staged verified epoch-1 input on {host}", flush=True)


def launch(config: dict, host: str, gpu: int, shards: list[int]) -> int:
    python, features = host_settings(config, host)
    if not shards or len(shards) != len(set(shards)) or any(not 0 <= s < NUM_SHARDS for s in shards):
        raise ValueError(f"Invalid shard assignment: {shards}")
    csv = ",".join(map(str, shards))
    log = BASE / "epoch1_work_32" / f"launcher_{shards[0]:02d}.log"
    environment = (f"HITPH_PYTHON={shlex.quote(python)} "
                   f"HITPH_FEATURES_DIR={shlex.quote(features)} "
                   f"HITPH_ACCELERATED={int(bool(config.get('epoch1_accelerated')))}")
    if host == "Tang3":
        session = f"hitph_epoch1_{'accel' if config.get('epoch1_accelerated') else 'torch'}_{shards[0]:02d}"
        payload = (f"env {environment} bash {shlex.quote(str(REMOTE_WORKER))} "
                   f"{gpu} {shlex.quote(csv)} > {shlex.quote(str(log))} 2>&1")
        subprocess.run(["tmux", "new-session", "-d", "-s", session, payload], check=True)
        pid = int(subprocess.check_output(
            ["tmux", "display-message", "-p", "-t", session, "#{pane_pid}"], text=True).strip())
        print(f"Launched {host} GPU {gpu} shard(s) {csv}: PID {pid}; tmux {session}", flush=True)
        return pid
    command = (f"mkdir -p {shlex.quote(str(log.parent))}; "
               f"nohup env {environment} bash {shlex.quote(str(REMOTE_WORKER))} "
               f"{gpu} {shlex.quote(csv)} > {shlex.quote(str(log))} 2>&1 < /dev/null & echo $!")
    output = run_host(host, command)
    pid = int(output.splitlines()[-1])
    print(f"Launched {host} GPU {gpu} shard(s) {csv}: PID {pid}", flush=True)
    return pid


def gpu_slots(host: str) -> list[tuple[int, int, int]]:
    command = "nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits"
    rows = []
    for line in run_host(host, command).splitlines():
        gpu, free, util = (int(x.strip()) for x in line.split(","))
        if free >= 8192:
            rows.append((gpu, free, util))
    return rows


def metadata_done(features: Path, shard: int) -> bool:
    suffix = f"shard_{shard:02d}_of_{NUM_SHARDS:02d}"
    return ((features / f"index.{suffix}.json").is_file()
            and (features / f"extraction_stats.{suffix}.json").is_file())


def save_config(path: Path, config: dict) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(config, indent=2) + "\n")
    temporary.replace(path)


def process_alive(host: str, pid: int) -> bool:
    try:
        state = run_host(host, f"ps -p {pid} -o stat=")
    except subprocess.CalledProcessError:
        return False
    return bool(state and not state.startswith("Z"))


def replacement_gpu(entry: dict, active: list[dict], features: Path) -> tuple[str, int]:
    """Move a failed shard away from its previous GPU when memory is tight."""
    choices = []
    for host in HOSTS:
        for gpu, free, utilization in gpu_slots(host):
            if free < 12288 or (host, gpu) == (entry["host"], entry["gpu"]):
                continue
            assigned = sum(
                other is not entry and other["host"] == host and other["gpu"] == gpu
                and any(not metadata_done(features, shard) for shard in other["shards"])
                for other in active
            )
            choices.append((assigned, -free, utilization, host, gpu))
    if not choices:
        raise RuntimeError(f"No alternate GPU has 12 GiB free for shards {entry['shards']}")
    _assigned, _free, _utilization, host, gpu = min(choices)
    return host, gpu


def finish(config_path: Path, config: dict, input_path: Path,
           features: Path, max_new_workers: int) -> None:
    from merge_shards import merge

    # Initial workers cover 0..5. They may still be running when the permanent
    # cache finishes; subsequent workers own only 6..31.
    state = config["epoch1_distributed"]
    initial = state["initial_workers"]
    if [item["shards"][0] for item in initial] != list(range(len(INITIAL))):
        raise RuntimeError("Initial epoch-1 shard assignments differ from the coordinator")
    post = state.setdefault("post_permanent_workers", [])
    claimed = [shard for entry in post for shard in entry["shards"]]
    if len(claimed) != len(set(claimed)) or any(shard < len(INITIAL) or shard >= NUM_SHARDS for shard in claimed):
        raise RuntimeError("Post-permanent epoch-1 shard assignments overlap")
    remaining = [shard for shard in range(len(INITIAL), NUM_SHARDS) if shard not in claimed]
    slots = []
    for host in HOSTS:
        for gpu, free, util in gpu_slots(host):
            slots.append((util, -free, host, gpu))
    slots.sort()
    if remaining and not slots:
        raise RuntimeError("No GPU has 8 GiB free for the remaining epoch-1 shards")
    selected = slots[: min(max_new_workers, len(slots), len(remaining))]
    assignments = [[] for _ in selected]
    for index, shard in enumerate(remaining):
        assignments[index % len(selected)].append(shard)
    for (_util, _free, host, gpu), shards in zip(selected, assignments):
        entry = {"host": host, "gpu": gpu, "shards": shards,
                 "pid": launch(config, host, gpu, shards)}
        post.append(entry)
        save_config(config_path, config)

    start = time.monotonic()
    retry_count: dict[tuple[int, ...], int] = {}
    while True:
        missing = [s for s in range(NUM_SHARDS) if not metadata_done(features, s)]
        if not missing:
            break
        for entry in initial + post:
            unfinished = [s for s in entry["shards"] if s in missing]
            if not unfinished or process_alive(entry["host"], int(entry["pid"])):
                continue
            key = tuple(entry["shards"])
            retry_count[key] = retry_count.get(key, 0) + 1
            if retry_count[key] > 3:
                raise RuntimeError(f"Epoch-1 worker failed repeatedly: {entry}")
            host, gpu = replacement_gpu(entry, initial + post, features)
            print(f"Restarting stopped epoch-1 worker for shards {entry['shards']} "
                  f"on {host} GPU {gpu}", flush=True)
            entry["host"], entry["gpu"] = host, gpu
            entry["pid"] = launch(config, entry["host"], entry["gpu"], entry["shards"])
            save_config(config_path, config)
        print(f"Waiting for {len(missing)} epoch-1 shards; first missing: {missing[0]}; "
              f"elapsed {round((time.monotonic()-start)/3600, 1)} h", flush=True)
        time.sleep(120)
    result = merge(input_path, features, NUM_SHARDS)
    if result["samples_total"] != 158930:
        raise RuntimeError(f"Wrong epoch-1 cache size: {result['samples_total']}")
    print("Published epoch-1 negative index:", json.dumps(result), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("start", "finish"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/hitph_protenix_cycle4_nomsa.json")
    parser.add_argument("--input-json", type=Path, default=ROOT / "work/hitph_cycle4_nomsa/train_negatives_epoch_0001.json")
    parser.add_argument("--max-new-workers", type=int, default=18)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    features = Path(config["output_root"]) / "features"
    if args.mode == "start":
        if config.get("epoch1_distributed"):
            raise RuntimeError("Epoch-1 distributed workers are already registered in the config")
        stage(config, args.input_json)
        launched = []
        for shard, (host, gpu) in enumerate(INITIAL):
            launched.append({"host": host, "gpu": gpu, "shards": [shard],
                             "pid": launch(config, host, gpu, [shard])})
        config["epoch1_distributed"] = {"num_shards": NUM_SHARDS,
                                        "input_sha256": hashlib.sha256(args.input_json.read_bytes()).hexdigest(),
                                        "initial_workers": launched,
                                        "remaining_started_after_permanent": True}
        save_config(args.config, config)
    else:
        if config.get("epoch1_distributed", {}).get("auto_restart") is False:
            raise RuntimeError("Epoch-1 extraction was stopped; refusing to restart workers")
        expected = config.get("epoch1_distributed", {}).get("input_sha256")
        if expected != hashlib.sha256(args.input_json.read_bytes()).hexdigest():
            raise RuntimeError("Epoch-1 input differs from the early distributed cache")
        finish(args.config, config, args.input_json, features, args.max_new_workers)


if __name__ == "__main__":
    main()
