# Fused residual RMSNorm on Turing

Original CUDA C++ implementations of

```text
z = float32(x) + float32(residual)
y = cast_input_dtype(z * rsqrt(mean(z*z) + eps) * float32(weight))
```

The residual sum stays in FP32 even for FP16 input. This matters: rounding the sum to FP16 before normalization changes the operation. Output has the input dtype, weight has the same dtype, and epsilon must be finite and positive. This operation produces the normalized output only; it does not update the residual buffer.

## Implementation

Each block owns one contiguous row. Threads read adjacent columns, reduce their squared sums with full-warp shuffles, then use a small shared-memory array for the block reduction. All threads participate, including threads beyond the row tail. The output pass reloads the inputs instead of retaining an unbounded register array.

- `fused256`: fixed 256-thread baseline, one launch.
- `width_dispatch`: 32 threads for widths up to 128, 128 threads up to 1024, otherwise 256. This is a candidate dispatch policy, not a universal performance guarantee.
- `two_kernel`: FP32 residual addition into scratch, followed by normalization. It retains the same numerical semantics and adds an intermediate buffer and launch.

`kernels.cu` exposes a C ABI. `validate.py` connects it to contiguous PyTorch CUDA tensors through ctypes and submits work on PyTorch's current stream. It is an inference harness, without autograd registration or a packaged PyTorch operator. The caller must keep buffers alive until stream completion, supply nonoverlapping buffers, and provide FP32 scratch for the separate-kernel baseline. General strided tensors are outside the contract.

## Reproduce

Use a T4 GPU runtime and run `Fused-Residual-RMSNorm.ipynb`, or run these commands in a CUDA environment with PyTorch:

```bash
nvcc -std=c++17 -O3 -arch=sm_75 --fmad=false -shared -Xcompiler -fPIC kernels.cu -o librmsnorm.so
python validate.py
nvcc -std=c++17 -O3 -arch=sm_75 --fmad=false native.cu -o native
./native
compute-sanitizer --tool memcheck --error-exitcode 1 ./native
compute-sanitizer --tool synccheck --error-exitcode 1 ./native
./native benchmark
```

The compiler intentionally disables multiply-add contraction to keep the expression order consistent across comparators. Fast-math is not enabled. The initial target is SM75; other architectures require their own build and validation.

## Evidence and timing boundaries

Executed on a Google Colab Tesla T4 (SM75), driver 580.82.07, nvcc 12.8.93, PyTorch 2.11.0+cu128 and Python 3.13.15. Raw logs and all timing groups are included. The executed [Colab notebook](https://colab.research.google.com/drive/1m3XEtL3jRPbEKH_-ML28TDqaonUsMR5p) is also saved as a portable notebook in this repository.

The Python suite passed 306 input cases across both dtypes against double-precision arithmetic: 17 widths, three row counts and three magnitude/epsilon pairs. Each case exercises all three variants, giving 918 numerical checks. Another 18 nonfinite checks compare IEEE propagation with the equivalent PyTorch expression. The native harness independently passed 102 checks with deterministic signed inputs and poison initialization. Compute Sanitizer memcheck and synccheck both reported zero errors for that native suite. Sanitizer coverage is the native suite, not every Python test.

Selected C++ submission medians, in microseconds:

| Type | Rows × width | Fused 256 | Width dispatch | Two kernels |
|---|---:|---:|---:|---:|
| FP32 | 32 × 768 | 4.58 | 5.23 | 7.69 |
| FP16 | 32 × 768 | 4.44 | 3.78 | 7.90 |
| FP32 | 1024 × 4096 | 305.70 | 305.59 | 371.07 |
| FP16 | 1024 × 4096 | 123.39 | 123.55 | 259.37 |
| FP32 | 1024 × 4099 | 350.36 | 350.31 | 397.99 |
| FP16 | 1024 × 4099 | 141.53 | 141.51 | 287.73 |

Fusion reduced latency against the separate-kernel baseline for these cases. At FP16 1024 × 4096, the median ratio is 2.10×. The dispatch candidate regresses FP32 32 × 768, and both 128 × 1024 cases also favored fixed 256 threads in this run. I am keeping the dispatch experimental rather than selecting it as the default. Small-case measurements include noisy groups; the complete `native-timing.log` makes those visible. Eager PyTorch comparisons in `results.json` have a different submission path and must not be mixed into this native table.

Nsight Compute basic profiling succeeded. For the FP16 fused 1024 × 4096 workload, it reported 69.03% DRAM throughput, 50.22% compute throughput and 96.56% achieved occupancy. Its replay duration was 150.08 µs with an SM clock around 585 MHz; this is a separate profiled execution, not the benchmark median. The report points toward memory traffic as a next investigation, but does not prove a vectorization speedup. The three-row width-one probe underutilized the grid, illustrating why occupancy conclusions need a concrete workload. Raw profiles are `ncu-fused-fp16.log`, `ncu-fused-fp32.log`, `ncu-two-kernel-fp16.log` and `ncu.log`.

Both timing paths use CUDA events with 30 warmups, seven groups of 200 submissions, and preserve every group. C++ submission removes Python wrapper overhead but short kernels may still be limited by host launch throughput. Python timing includes wrapper validation, ctypes calls and possible submission gaps; it is not isolated kernel latency or wall-clock end-to-end latency. The eager PyTorch comparator includes its intermediate allocations and FP32 conversions. It is an equivalent expression, not an optimized vendor RMSNorm kernel. Inputs are reused, so these are warm-cache repeated-workload measurements. Results need review for spread, regressions and fixed-order effects before choosing a dispatch policy.

## Attribution

Developed with Codex assistance. Kernel source is original to this project; CUDA and PyTorch provide the runtime and reference operations. This is an initial implementation milestone in a broader GPU inference portfolio. Framework operator registration, analysis of memory transactions, vectorized access and controlled benchmark ordering remain future work. No RTX 2060 or other-architecture validation has been executed.
