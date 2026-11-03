#!/usr/bin/env python3
"""Exercise the production head on variable-length inputs under compute-sanitizer."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from foldabs_repro.tmp_prediction.model import TMPTrunkSpecificityHead
from foldabs_repro.tmp_prediction.interfaces import REQUIRED_ROLES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[418, 416, 419])
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    head = TMPTrunkSpecificityHead(c_s=384, c_z=128, c_dist=64, pairformer_blocks=4,
        pooling_mode="C", hidden_dim=512, dropout=0.1, triangle_multiplicative="cuequivariance",
        triangle_attention="cuequivariance").cuda().train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=0.01)
    for repeat in range(args.repeats):
        for tokens in args.tokens:
            optimizer.zero_grad(set_to_none=True)
            single = torch.randn(tokens, 384, device="cuda").half().float()
            pair = torch.randn(tokens, tokens, 128, device="cuda").half().float()
            probability = torch.randn(tokens, tokens, 64, device="cuda").softmax(-1).half().float()
            lengths = [114, tokens - 114 - 9 - 182, 9, 182]
            asym = torch.repeat_interleave(torch.arange(4, device="cuda"), torch.tensor(lengths, device="cuda"))
            print(json.dumps({"repeat": repeat, "tokens": tokens, "stage": "forward"}), flush=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = head(single, pair, probability, asym, REQUIRED_ROLES).float()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, torch.ones_like(logit))
            print(json.dumps({"repeat": repeat, "tokens": tokens, "stage": "backward"}), flush=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), float("inf"), error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            print(json.dumps({"repeat": repeat, "tokens": tokens, "stage": "done", "loss": loss.item(),
                              "gradient_norm": norm.item()}), flush=True)
            del single, pair, probability, asym, logit, loss


if __name__ == "__main__":
    main()
