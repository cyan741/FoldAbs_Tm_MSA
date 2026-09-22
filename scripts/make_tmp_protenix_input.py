#!/usr/bin/env python3
"""Convert paired-alpha/beta MHC-I CSV rows into Protenix complex inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from foldabs_repro.tmp_prediction.data import TMPDatasetSchema, clean_sequence, load_hla_sequences, load_tmp_csv
from foldabs_repro.tmp_prediction.input_builder import build_protenix_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--hla-sequences-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--beta2m-sequence", default="")
    parser.add_argument("--require-label", action="store_true")
    parser.add_argument("--require-split", action="store_true")
    parser.add_argument("--id-column", default="ID")
    parser.add_argument("--tcra-column", default="TCRa")
    parser.add_argument("--tcrb-column", default="TCRb")
    parser.add_argument("--peptide-column", default="Peptide")
    parser.add_argument("--hla-column", default="HLA")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--split-column", default="split")
    args = parser.parse_args()

    schema = TMPDatasetSchema(
        id_column=args.id_column,
        tcra_column=args.tcra_column,
        tcrb_column=args.tcrb_column,
        peptide_column=args.peptide_column,
        hla_column=args.hla_column,
        label_column=args.label_column,
        split_column=args.split_column,
    )
    frame = load_tmp_csv(
        args.input_csv,
        schema,
        require_label=args.require_label,
        require_split=args.require_split,
    )
    hla_sequences = load_hla_sequences(args.hla_sequences_json)
    beta2m = clean_sequence(args.beta2m_sequence)
    records, manifest = build_protenix_inputs(frame, hla_sequences, beta2m_sequence=beta2m)

    json_path = Path(args.output_json)
    manifest_path = Path(args.output_manifest)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest.to_csv(manifest_path, index=False)
    print(f"Wrote {len(records)} TMP complexes to {json_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
