"""Correctness and repeated CUDA-event timings for equivalent FP32-sum semantics."""
import ctypes
import json
import platform
import statistics
from pathlib import Path

import torch


class Kernel:
    def __init__(self, library="./librmsnorm.so"):
        self.lib = ctypes.CDLL(str(Path(library).resolve()))
        self.lib.rmsnorm.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 2 + [ctypes.c_float] + [ctypes.c_int] * 2 + [ctypes.c_void_p]
        self.lib.rmsnorm.restype = ctypes.c_int
        self.lib.rmsnorm_error.argtypes = [ctypes.c_int]
        self.lib.rmsnorm_error.restype = ctypes.c_char_p

    def launch(self, x, residual, weight, out, scratch, eps, variant):
        if x.ndim != 2 or x.dtype not in (torch.float16, torch.float32):
            raise ValueError("x must be a two-dimensional FP16 or FP32 tensor")
        rows, width = x.shape
        if residual.shape != x.shape or out.shape != x.shape or weight.shape != (width,):
            raise ValueError("incompatible residual, weight, or output shape")
        buffers = (x, residual, weight, out, scratch)
        if any(not t.is_cuda or not t.is_contiguous() or t.device != x.device for t in buffers):
            raise ValueError("buffers must be contiguous on the same CUDA device")
        if any(t.dtype != x.dtype for t in buffers[:4]) or scratch.dtype != torch.float32 or scratch.shape != x.shape:
            raise ValueError("incompatible buffer dtype or scratch shape")
        if out.data_ptr() in (x.data_ptr(), residual.data_ptr(), weight.data_ptr()):
            raise ValueError("output must not alias inputs")
        # Fresh allocations in the harness satisfy the stronger C ABI no-overlap contract.
        with torch.cuda.device(x.device):
            error = self.lib.rmsnorm(*(t.data_ptr() for t in buffers), rows, width, eps,
                                     int(x.dtype == torch.float16), variant,
                                     torch.cuda.current_stream(x.device).cuda_stream)
        if error:
            raise RuntimeError(self.lib.rmsnorm_error(error).decode())


def reference(x, residual, weight, eps):
    z = x.float() + residual.float()
    return (z * torch.rsqrt(z.square().mean(dim=-1, keepdim=True) + eps) * weight.float()).to(x.dtype)


def timing(fn, repeats=200, groups=7):
    for _ in range(30):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(groups):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / repeats)
    return {"median_us": statistics.median(samples), "min_us": min(samples),
            "max_us": max(samples), "samples_us": samples}


def main():
    torch.manual_seed(3487)
    kernel = Kernel()
    env = {"gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
           "torch": torch.__version__, "torch_cuda": torch.version.cuda, "python": platform.python_version()}
    print(json.dumps({"environment": env}), flush=True)
    cases = []
    for dtype in (torch.float32, torch.float16):
        for rows in (1, 3, 129):
            for width in (1, 7, 31, 32, 33, 127, 128, 129, 255, 256, 257, 1023, 1024, 1025, 4096, 4099, 8192):
                for magnitude, eps in ((1.0, 1e-5), (1e-4, 1e-8), (0.0, 1e-5)):
                    x = (torch.randn(rows, width, device="cuda") * magnitude).to(dtype)
                    r = (torch.randn_like(x.float()) * magnitude).to(dtype)
                    w = torch.randn(width, device="cuda", dtype=dtype)
                    out, scratch = torch.empty_like(x), torch.empty_like(x, dtype=torch.float32)
                    # Independent double-precision arithmetic checks accumulation and rounding.
                    z = x.double() + r.double()
                    expected = (z * torch.rsqrt(z.square().mean(-1, keepdim=True) + eps) * w.double()).to(dtype)
                    for variant in (0, 1, 2):
                        kernel.launch(x, r, w, out, scratch, eps, variant)
                        torch.testing.assert_close(out, expected, rtol=2e-5 if dtype == torch.float32 else 2e-3,
                                                   atol=2e-6 if dtype == torch.float32 else 2e-3)
                    cases.append([str(dtype), rows, width, magnitude, eps])
        # IEEE nonfinite propagation should match the equivalent framework operation.
        for special in (float("nan"), float("inf"), -float("inf")):
            x = torch.ones(3, 33, device="cuda", dtype=dtype)
            x[0, 0] = special
            r, w = torch.zeros_like(x), torch.ones(33, device="cuda", dtype=dtype)
            out, scratch = torch.empty_like(x), torch.empty_like(x, dtype=torch.float32)
            expected = reference(x, r, w, 1e-5)
            for variant in (0, 1, 2):
                kernel.launch(x, r, w, out, scratch, 1e-5, variant)
                torch.testing.assert_close(out, expected, equal_nan=True)
    torch.cuda.synchronize()
    print(json.dumps({"correctness_input_cases": len(cases), "variant_checks": len(cases) * 3,
                      "nonfinite_checks": 18, "status": "PASS"}), flush=True)
    results = []
    for dtype in (torch.float32, torch.float16):
        for rows, width in ((1, 128), (1, 4096), (32, 768), (128, 1024), (1024, 4096), (1024, 4099), (256, 8192)):
            x = torch.randn(rows, width, device="cuda", dtype=dtype)
            r, w = torch.randn_like(x), torch.randn(width, device="cuda", dtype=dtype)
            out, scratch = torch.empty_like(x), torch.empty_like(x, dtype=torch.float32)
            row = {"dtype": str(dtype), "rows": rows, "width": width, "timing": {}}
            for variant, name in ((0, "fused256"), (1, "width_dispatch"), (2, "two_kernel")):
                row["timing"][name] = timing(lambda v=variant: kernel.launch(x, r, w, out, scratch, 1e-5, v))
            row["timing"]["pytorch_eager"] = timing(lambda: reference(x, r, w, 1e-5))
            results.append(row)
            print(json.dumps(row), flush=True)
    Path("results.json").write_text(json.dumps({"environment": env, "cases": cases, "benchmarks": results,
                                               "warmup": 30, "repeats": 200, "groups": 7}, indent=2))


if __name__ == "__main__":
    main()
