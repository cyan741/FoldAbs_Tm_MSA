from __future__ import annotations

import json
import sys

import torch


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot see the CUDA GPU.")

    device = torch.device("cuda:0")
    capability = torch.cuda.get_device_capability(device)
    properties = torch.cuda.get_device_properties(device)
    arch_list = torch.cuda.get_arch_list()

    a = torch.randn((1024, 1024), device=device, dtype=torch.bfloat16)
    b = torch.randn((1024, 1024), device=device, dtype=torch.bfloat16)
    c = a @ b
    torch.cuda.synchronize(device)

    report = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "gpu": properties.name,
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "total_memory_gib": round(properties.total_memory / 1024**3, 2),
        "compiled_architectures": arch_list,
        "bf16_matmul_shape": list(c.shape),
        "bf16_result_finite": bool(torch.isfinite(c).all().item()),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if capability[0] >= 12 and "sm_120" not in arch_list:
        raise RuntimeError(
            "This PyTorch wheel does not contain sm_120 support for RTX 5090."
        )
    if not report["bf16_result_finite"]:
        raise RuntimeError("The BF16 CUDA matrix multiplication returned non-finite data.")


if __name__ == "__main__":
    main()

