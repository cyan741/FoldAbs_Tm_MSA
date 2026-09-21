#!/usr/bin/env python3
"""Run the established extractor with verified local MSAs and no outbound sockets."""
from __future__ import annotations
import json
from pathlib import Path
import runpy
import socket
import sys


def validate_input(path):
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not records:
        raise ValueError("Empty local MSA manifest")
    for record in records:
        for item in record["sequences"]:
            chain = item["proteinChain"]
            if "pairedMsaPath" in chain:
                raise ValueError("pairedMsaPath must be absent for this unpaired-only experiment")
            msa = Path(chain.get("unpairedMsaPath", ""))
            if not msa.is_file() or msa.stat().st_size == 0:
                raise ValueError(f"Missing local MSA: {msa}")
            with msa.open(encoding="utf-8") as handle:
                if not handle.readline().startswith(">"):
                    raise ValueError(f"Invalid A3M header: {msa}")
                sequence = []
                for line in handle:
                    if line.startswith(">"):
                        break
                    sequence.append(line.strip())
            if "".join(sequence).replace("-", "").replace(".", "").upper() != chain["sequence"]:
                raise ValueError(f"Local A3M query mismatch: {msa}")


def install_network_guard():
    # Local IPC remains available to PyTorch; external network connection
    # attempts (including any remote MSA fallback) raise before connecting.
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def checked(original):
        def connect(sock, address):
            if sock.family in (socket.AF_INET, socket.AF_INET6):
                host = address[0]
                if host not in ("127.0.0.1", "::1", "localhost"):
                    raise RuntimeError(f"LOCAL_MSA_ONLY: outbound network blocked: {address}")
            return original(sock, address)
        return connect

    socket.socket.connect = checked(original_connect)
    socket.socket.connect_ex = checked(original_connect_ex)


def main():
    if "--input-json" not in sys.argv or "--use-msa" not in sys.argv:
        raise ValueError("Local extraction requires --input-json PATH and --use-msa")
    path = sys.argv[sys.argv.index("--input-json")+1]
    validate_input(path)
    install_network_guard()
    print("LOCAL_MSA_ONLY: input validated; outbound connections blocked", flush=True)
    target = Path(__file__).with_name("extract_protenix_embeddings.py")
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
