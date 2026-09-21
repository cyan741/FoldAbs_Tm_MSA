"""Offline checks for MSA conversion, cache reuse, selection and reporting."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_tm_local_uniref50_comparison as pipeline


class PipelineTests(unittest.TestCase):
    def test_stockholm_dots_insertions_deletions_and_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sto = root / "a.sto"
            sto.write_text("# STOCKHOLM 1.0\nquery AC.D-E\nhit1 ACXDYE\nhit2 A..D-E\n//\n")
            output = root / "a.a3m"
            self.assertEqual(pipeline.convert_alignment(sto, output, "ACDE", 1), 2)
            self.assertEqual(output.read_text(), ">query\nACDE\n>hit1\nACxDyE\n")
            with self.assertRaises(ValueError):
                pipeline.convert_alignment(sto, output, "AAAA", 1)

    def test_cache_resume_preserves_search_time_and_repairs_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def fake_search(command, **kwargs):
                output = Path(command[command.index("-A") + 1])
                output.write_text("# STOCKHOLM 1.0\nquery ACDE\nhit ACDE\n//\n")
                class Result:
                    returncode = 0
                return Result()
            with patch.object(pipeline.subprocess, "run", side_effect=fake_search) as mocked:
                initial = pipeline.search(["ACDE", "ACDE"], root / "parent", root / "cache", "hmmer", 8)
                seconds = initial[pipeline.key("ACDE")]["search_seconds"]
                again = pipeline.search(["ACDE"], root / "parent", root / "cache", "hmmer", 8)
                self.assertEqual(mocked.call_count, 1)
                self.assertEqual(seconds, again[pipeline.key("ACDE")]["search_seconds"])
                Path(again[pipeline.key("ACDE")]["a3m"]).write_text("corrupt")
                pipeline.search(["ACDE"], root / "parent", root / "cache", "hmmer", 8)
                self.assertEqual(mocked.call_count, 2)

    def test_export_keeps_query_and_limits_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a3m = root / "hits.a3m"
            a3m.write_text(">query\nACDE\n" + "".join(f">h{i}\nACDE\n" for i in range(5000)))
            records = [{"name": "a", "sequences": [{"proteinChain": {"sequence": "ACDE"}}]}]
            cache = {pipeline.key("ACDE"): {"a3m": str(a3m)}}
            pipeline.export_inputs(records, cache, root / "input.json")
            self.assertEqual((root / "input_depth4096.a3m").read_text().count(">"), 4097)
            self.assertNotIn("unpairedMsaPath", records[0]["sequences"][0]["proteinChain"])
            self.assertNotIn("pairedMsaPath", pipeline.read(root / "input.json")[0]["sequences"][0]["proteinChain"])
            self.assertEqual(pipeline.read(root / "input.json")[0]["name"], "a")

    def test_verify_input_rejects_paired_msa_field(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a3m = root / "x.a3m"
            a3m.write_text(">query\nACDE\n", encoding="utf-8")
            manifest = root / "input.json"
            pipeline.write(manifest, [{"sequences": [{"proteinChain": {
                "unpairedMsaPath": str(a3m), "pairedMsaPath": ""}}]}])
            with self.assertRaises(RuntimeError):
                pipeline.verify_unpaired_only_input(manifest)

    def test_training_preserves_tm_config(self):
        args = pipeline.training_arguments("data", "split", "esm", "feat", "out", 46, "local")
        values = dict(zip(args[::2], args[1::2]))
        self.assertEqual(values["--paircond-mode"], "additive")
        self.assertEqual(values["--ema-decay"], "0.995")
        self.assertNotIn("--pre-pairformer-layernorm", args)
        self.assertNotIn("--structural-component-layernorm", args)
        self.assertEqual(values["--seed"], "46")

    def test_summary_pairs_matching_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {m: {s: root / f"{m}{s}" for s in range(42, 48)} for m in ("regular", "compact")}
            for m in paths:
                for s in range(42, 48):
                    pipeline.write(paths[m][s] / "metrics.json", {"metrics": {"test": {
                        "spearman_rho": .8 + (s - 42) / 1000 + (.01 if m == "compact" else 0), "mae": 5}}})
            timing = dict(target_queries=610, regular_search_seconds=100, compact_search_seconds=10,
                          search_ratio=10, build_seconds=20, compact_cold_seconds=31, regular_route_seconds=102)
            pipeline.summary(root, paths, timing)
            report = pipeline.read(root / "summary.json")
            self.assertEqual(len(report["rows"]), 6)
            self.assertAlmostEqual(report["aggregates"]["compact"]["rho_mean"] - report["aggregates"]["regular"]["rho_mean"], .01)


if __name__ == "__main__":
    unittest.main()
