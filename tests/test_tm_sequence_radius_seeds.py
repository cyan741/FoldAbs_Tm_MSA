#!/usr/bin/env python3
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import select_tm_sequence_radius_seeds as selector


class Tests(unittest.TestCase):
    sequences = ["ACDEFG", "ACDEFA", "ACDYYY", "YYYYYY", "ACDEFGHIK", "KLMNPQ", "ACDE"]

    def test_words_and_distance(self):
        self.assertEqual(selector.kmers("ACDEF"), {"ACD", "CDE", "DEF"})
        words = {s: selector.kmers(s) for s in self.sequences}
        for a in self.sequences:
            self.assertEqual(selector.distance(a, a, words, 5), 0)
            for b in self.sequences:
                self.assertAlmostEqual(selector.distance(a, b, words, 5), selector.distance(b, a, words, 5))

    def test_radius_and_monotonicity(self):
        order, curve = selector.traversal(self.sequences)
        self.assertEqual((order, curve), selector.traversal(list(reversed(self.sequences))))
        words = {s: selector.kmers(s) for s in self.sequences}
        span = max(map(len, order)) - min(map(len, order))
        for row in curve:
            n = row["seed_count"]
            expected = max(min(selector.distance(s, seed, words, span) for seed in order[:n]) for s in order)
            self.assertAlmostEqual(row["radius"], expected)
        counts = [selector.stop_count(curve, r) for r in (0, .2, .4, .8, 1)]
        self.assertEqual(counts, sorted(counts, reverse=True))
        self.assertEqual(curve[-1]["radius"], 0)

    def test_edge_cases(self):
        self.assertEqual(selector.traversal([]), ([], []))
        self.assertEqual(selector.stop_count([], .2), 0)
        _, curve = selector.traversal(["AAA", "AAA"])
        self.assertEqual(selector.stop_count(curve, 0), 1)
        for invalid in (float("nan"), float("inf"), -1, 2):
            with self.assertRaises(ValueError):
                selector.stop_count(curve, invalid)

    def test_sequence_only_end_to_end_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "split.csv"
            def make_manifest(label, heldout):
                with manifest.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["split", "heavy", "light", "label"])
                    writer.writeheader()
                    for s in self.sequences:
                        writer.writerow(dict(split="train", heavy=s, light="", label=label))
                    writer.writerow(dict(split="train", heavy="ACDEFG", light="KLMNPQ", label=label))
                    writer.writerow(dict(split="test", heavy=heldout, light="", label=label))
            make_manifest(1, "VVVVVV")
            first = selector.run(manifest, root / "first", [.2, .4])
            make_manifest(999, "WWWWWW")
            second = selector.run(manifest, root / "second", [.2, .4])
            self.assertEqual(first["exploratory_summaries"], second["exploratory_summaries"])
            for r in (.2, .4):
                one = json.loads((root / "first" / f"radius_{r}" / "selected_seeds.json").read_text())
                two = json.loads((root / "second" / f"radius_{r}" / "selected_seeds.json").read_text())
                self.assertEqual(one, two)
                self.assertNotIn("VVVVVV", {row["sequence"] for row in one})
            with self.assertRaises(FileExistsError):
                selector.run(manifest, root / "first", [.2])


if __name__ == "__main__":
    unittest.main()
