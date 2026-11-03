#!/usr/bin/env python3
"""Replay online training in a separate output directory with CUDA stage tracing."""

import functools
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_mini_esm_online_ddp as training


def main():
    synchronize = os.environ.get("DEBUG_SYNC", "1") == "1"
    if synchronize and os.environ.get("CUDA_LAUNCH_BLOCKING") != "1":
        raise RuntimeError("Set CUDA_LAUNCH_BLOCKING=1 before starting the diagnostic process")
    context = {"rank": int(os.environ.get("RANK", 0)), "sample_id": None}

    def traced(name, function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            if name == "encoder":
                context["sample_id"] = args[2]
            print(json.dumps({**context, "stage": name, "state": "start", "time": time.time()}), flush=True)
            result = function(*args, **kwargs)
            if synchronize:
                training.torch.cuda.synchronize()
            print(json.dumps({**context, "stage": name, "state": "done", "time": time.time()}), flush=True)
            return result
        return wrapped

    training.OnlineFeatures.infer = traced("encoder", training.OnlineFeatures.infer)
    training.forward = traced("head_forward", training.forward)
    training.torch.Tensor.backward = traced("backward", training.torch.Tensor.backward)
    training.torch.optim.AdamW.step = traced("optimizer", training.torch.optim.AdamW.step)
    training.main()


if __name__ == "__main__":
    main()
