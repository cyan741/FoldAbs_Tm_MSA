#!/usr/bin/env python3
"""Small standalone triangle-attention backward probe for compute-sanitizer."""

import argparse
import json
import torch
from cuequivariance_torch.primitives.triangle import triangle_attention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[416])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mask-mode", choices=["ones", "none"], default="ones")
    parser.add_argument("--compare-reference", action="store_true")
    args = parser.parse_args()
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    for iteration in range(args.repeats):
        for tokens in args.tokens:
            shape = (1, tokens, 4, tokens, 32)
            inputs = [torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                      for _ in range(3)]
            bias = torch.randn((1, 1, 4, tokens, tokens), device="cuda", requires_grad=True)
            mask = (torch.ones((1, tokens, 1, 1, tokens), device="cuda", dtype=torch.bool)
                    if args.mask_mode == "ones" else None)
            print(json.dumps({"iteration": iteration, "tokens": tokens, "mask": args.mask_mode,
                              "stage": "forward"}), flush=True)
            output = triangle_attention(*inputs, bias, mask, scale=32 ** -0.5)
            print(json.dumps({"iteration": iteration, "tokens": tokens, "mask": args.mask_mode,
                              "stage": "backward"}), flush=True)
            gradient = torch.randn_like(output)
            output.backward(gradient)
            torch.cuda.synchronize()
            print(json.dumps({"iteration": iteration, "tokens": tokens,
                              "finite_gradients": all(bool(torch.isfinite(value.grad).all()) for value in [*inputs, bias])}), flush=True)
            if args.compare_reference:
                gradients = [value.grad.detach().clone() for value in [*inputs, bias]]
                for value in [*inputs, bias]:
                    value.grad = None
                reference = torch.nn.functional.scaled_dot_product_attention(
                    *inputs, attn_mask=bias, dropout_p=0.0, scale=32 ** -0.5)
                reference.backward(gradient)
                differences = [float((actual.float() - value.grad.float()).norm() / value.grad.float().norm())
                               for actual, value in zip(gradients, [*inputs, bias], strict=True)]
                print(json.dumps({"tokens": tokens, "reference": "torch_sdpa_all_valid",
                                  "output_max_abs_difference": float((reference.float() - output.float()).abs().max()),
                                  "gradient_relative_l2_q_k_v_bias": differences}), flush=True)
                del gradients, reference
            del inputs, bias, mask, output, gradient


if __name__ == "__main__":
    main()
