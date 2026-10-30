"""Exercise the installed GPU runtime without loading models or starting Ray.

Run in a fresh process after setting LD_LIBRARY_PATH and CUDA_VISIBLE_DEVICES.
Tensor inputs are small; CUDA/NCCL contexts and library workspaces can still
consume hundreds of MiB. This check does not validate multi-GPU training.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import gc
import os
from pathlib import Path
import tempfile
import traceback


def check_cuda(device: int) -> None:
    import torch

    # vLLM switches is_available() to an NVML-only discovery check. Require
    # CUDA runtime initialization and an actual allocation instead.
    os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "0"
    torch.cuda.init()
    torch.cuda.set_device(device)
    value = torch.ones(1, device="cuda")
    torch.cuda.synchronize()
    if value.item() != 1:
        raise RuntimeError("CUDA allocation/readback returned an unexpected value")
    print(f"torch={torch.__version__}, CUDA={torch.version.cuda}, device={torch.cuda.get_device_name(device)}")
    maps = Path("/proc/self/maps")
    if maps.exists():
        libraries = sorted({line.split()[-1] for line in maps.read_text().splitlines() if "libcuda.so" in line})
        print(f"loaded CUDA driver libraries: {libraries}")


def check_matmul() -> None:
    import torch

    left = torch.randn(128, 128, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    right = torch.randn_like(left, requires_grad=True)
    result = left @ right
    result.float().square().mean().backward()
    torch.cuda.synchronize()
    for name, value in (("result", result), ("left gradient", left.grad), ("right gradient", right.grad)):
        if value is None or not torch.isfinite(value).all().item():
            raise RuntimeError(f"BF16 matmul produced invalid {name}")


def check_flash_attention() -> None:
    import torch
    from flash_attn import flash_attn_func

    tensors = [torch.randn(1, 32, 2, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
               for _ in range(3)]
    result = flash_attn_func(*tensors, dropout_p=0.0, causal=True)
    result.float().square().mean().backward()
    torch.cuda.synchronize()
    if not torch.isfinite(result).all().item():
        raise RuntimeError("FlashAttention produced nonfinite output")
    if any(value.grad is None or not torch.isfinite(value.grad).all().item() for value in tensors):
        raise RuntimeError("FlashAttention produced missing/nonfinite gradients")


def check_cumem() -> None:
    import torch
    from vllm.device_allocator.cumem import CuMemAllocator

    os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "0"
    allocator = CuMemAllocator.get_instance()
    tensor = None
    try:
        with allocator.use_memory_pool(tag="runtime_check"):
            tensor = torch.full((1024,), 7.0, dtype=torch.float32, device="cuda")
        torch.cuda.synchronize()
        if allocator.get_current_usage() == 0:
            raise RuntimeError("Test allocation did not enter vLLM's CuMem pool")
        print(f"CuMem pool allocation: {allocator.get_current_usage()} bytes")
        allocator.sleep(offload_tags=("runtime_check",))
        allocator.wake_up(tags=["runtime_check"])
        torch.cuda.synchronize()
        if not torch.equal(tensor.cpu(), torch.full((1024,), 7.0)):
            raise RuntimeError("vLLM sleep/wake did not preserve tensor values")
    finally:
        # Free the tensor before the underlying pluggable allocator/pool.
        del tensor
        gc.collect()
        allocator.release_pools()


def check_nccl() -> None:
    import torch
    import torch.distributed as dist

    if dist.is_initialized():
        raise RuntimeError("Run this script in a fresh process with no existing process group")
    temporary_root = Path(os.environ.get("TMPDIR", str(Path(__file__).resolve().parents[1] / "vendor/runtime-checks")))
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nccl-check-", dir=temporary_root) as temporary:
        try:
            dist.init_process_group(
                backend="nccl", init_method=(Path(temporary) / "rendezvous").as_uri(),
                rank=0, world_size=1, timeout=timedelta(seconds=30),
                device_id=torch.device("cuda", torch.cuda.current_device()),
            )
            value = torch.tensor([3.0], device="cuda")
            dist.all_reduce(value)
            torch.cuda.synchronize()
            if value.item() != 3.0:
                raise RuntimeError("Single-rank NCCL all_reduce changed the value")
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="Logical index within CUDA_VISIBLE_DEVICES")
    parser.add_argument("--skip-nccl", action="store_true", help="Skip the single-rank NCCL check")
    args = parser.parse_args()
    os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "0"
    checks = [
        ("CUDA initialization and allocation", lambda: check_cuda(args.device)),
        ("BF16 matmul forward/backward", check_matmul),
        ("FlashAttention forward/backward", check_flash_attention),
        ("vLLM CuMem allocation and sleep/wake", check_cumem),
    ]
    if not args.skip_nccl:
        checks.append(("single-rank NCCL all_reduce", check_nccl))
    for name, check in checks:
        print(f"[RUN] {name}", flush=True)
        try:
            check()
        except Exception:
            print(f"[FAIL] {name}", flush=True)
            traceback.print_exc()
            raise SystemExit(1) from None
        print(f"[PASS] {name}", flush=True)
    print("GPU runtime smoke checks passed. Model execution and multi-GPU training remain untested.", flush=True)


if __name__ == "__main__":
    main()
