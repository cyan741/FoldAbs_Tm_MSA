from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_SCRIPT = PROJECT_ROOT / "scripts" / "build_compact_nb_msa_database.py"
CACHE_SCRIPT = PROJECT_ROOT / "scripts" / "prepare_compact_db_msa_cache.py"


def _key(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def test_compact_database_uses_only_cached_sequence_hits(tmp_path: Path) -> None:
    sequences = ["QVQLVESGGGLVQAGGSLRLSCAAS", "QVQLVESGGGLVQAGGSLRLSCAAT", "QVQLVESGGGLVQAGGSLRLSCAAV"]
    dataset = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "heavy": sequences,
            "light": ["", "", ""],
            "label": [60.0, 70.0, 80.0],
            "source": ["nbthermo_dsf_sypro"] * 3,
        }
    ).to_csv(dataset, index=False)
    cache_dir = tmp_path / "full_cache"
    cache_dir.mkdir()
    manifest = {}
    for index, sequence in enumerate(sequences):
        a3m = tmp_path / f"seed_{index}.a3m"
        a3m.write_text(f">query\n{sequence}\n>homolog\n{sequence}A\n", encoding="utf-8")
        manifest[_key(sequence)] = {"sequence": sequence, "unpairedMsaPath": str(a3m)}
    (cache_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    output_dir = tmp_path / "compact"
    subprocess.run(
        [
            sys.executable,
            str(DATABASE_SCRIPT),
            "--data-csv",
            str(dataset),
            "--msa-cache-dir",
            str(cache_dir),
            "--output-dir",
            str(output_dir),
            "--seed-count",
            "2",
            "--max-hits-per-seed",
            "0",
        ],
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
        check=True,
    )
    report = json.loads((output_dir / "compact_database_report.json").read_text(encoding="utf-8"))
    assert report["seed_count"] == 2
    assert report["unique_database_sequences"] >= 2
    assert (output_dir / "nb_compact_database.fasta").is_file()


def test_stockholm_conversion_marks_insertions_as_a3m_lowercase(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("compact_cache", CACHE_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stockholm = tmp_path / "alignment.sto"
    stockholm.write_text(
        "# STOCKHOLM 1.0\nquery AC-DE\nhit1 ACXDE\nhit2 AC--E\n//\n",
        encoding="utf-8",
    )
    a3m = tmp_path / "non_pairing.a3m"
    assert module._stockholm_to_a3m(stockholm, a3m, max_hits=0) == 3
    assert a3m.read_text(encoding="utf-8") == (
        ">query\nACDE\n>hit_00001\nACxDE\n>hit_00002\nAC-E\n"
    )
