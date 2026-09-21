#!/usr/bin/env python3
"""Extract frozen ESM-2 650M heavy/light embeddings for FoldAbS developability.

The FoldAbS paper mean-pools ESM-2 embeddings for the heavy and light chains
separately.  Nanobodies have no light chain; for those records we save an
all-zero light embedding and record ``has_light=False`` in the feature file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import esm
import torch

from foldabs_repro.data import DatasetSchema, load_fitness_csv


def _chunks(items: list[str], chunk_size: int):
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract frozen ESM-2 650M heavy/light embeddings."
    )
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--heavy-column", default="heavy")
    parser.add_argument("--light-column", default="light")
    parser.add_argument("--label-column", default="fitness")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    frame = load_fitness_csv(
        args.data_csv,
        schema=DatasetSchema(
            heavy_column=args.heavy_column,
            light_column=args.light_column,
            label_column=args.label_column,
        ),
    )
    if args.limit:
        frame = frame.iloc[: args.limit].reset_index(drop=True)

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(args.device)
    batch_converter = alphabet.get_batch_converter()
    embedding_dim = model.embed_dim

    unique_sequences = list(
        dict.fromkeys(
            sequence
            for row in frame.itertuples(index=False)
            for sequence in (row.heavy, row.light)
            if sequence
        )
    )
    sequence_embeddings: dict[str, torch.Tensor] = {}
    for batch_index, sequences in enumerate(_chunks(unique_sequences, args.batch_size), start=1):
        labels = [f"sequence_{index}" for index in range(len(sequences))]
        _, _, tokens = batch_converter(list(zip(labels, sequences)))
        result = model(tokens.to(args.device), repr_layers=[model.num_layers], return_contacts=False)
        representations = result["representations"][model.num_layers]
        for index, sequence in enumerate(sequences):
            # ESM adds a BOS token.  The end token is immediately after the sequence.
            sequence_embeddings[sequence] = (
                representations[index, 1 : len(sequence) + 1].mean(dim=0).float().cpu()
            )
        print(
            f"Embedded batch {batch_index}: {len(sequence_embeddings)}/{len(unique_sequences)} unique chains",
            flush=True,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    zero_light = torch.zeros(embedding_dim, dtype=torch.float16)
    index = []
    for row in frame.itertuples(index=False):
        output_path = output_dir / f"{row.sample_id}.pt"
        has_light = bool(row.light)
        torch.save(
            {
                "sample_id": row.sample_id,
                "e_h": sequence_embeddings[row.heavy].to(dtype=torch.float16),
                "e_l": (
                    sequence_embeddings[row.light].to(dtype=torch.float16)
                    if has_light
                    else zero_light.clone()
                ),
                "has_light": has_light,
            },
            output_path,
        )
        index.append({"sample_id": row.sample_id, "file": output_path.name, "has_light": has_light})
    (output_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Saved {len(index)} ESM-2 embeddings to {output_dir}")


if __name__ == "__main__":
    main()
