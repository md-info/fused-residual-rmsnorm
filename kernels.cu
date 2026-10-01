#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>

// All variants retain the residual sum in FP32 and round only the final output.
// Each block owns a row; every lane participates in the reduction, including tails.
template <class T> __device__ float read_value(T x) { return float(x); }
template <> __device__ float read_value<half>(half x) { return __half2float(x); }
template <class T> __device__ T write_value(float x) { return T(x); }
template <> __device__ half write_value<half>(float x) { return __float2half_rn(x); }

__device__ float warp_sum(float x) {
  for (int delta = 16; delta; delta >>= 1)
    x += __shfl_down_sync(0xffffffffu, x, delta);
  return x;
}

template <int Threads> __device__ float row_sum(float x) {
  __shared__ float partial[Threads / 32];
  __shared__ float total;
  x = warp_sum(x);
  if ((threadIdx.x & 31) == 0) partial[threadIdx.x / 32] = x;
  __syncthreads();
  if (threadIdx.x < 32) {
    x = threadIdx.x < Threads / 32 ? partial[threadIdx.x] : 0.0f;
    x = warp_sum(x);
    if (threadIdx.x == 0) total = x;
  }
  __syncthreads();
  return total;
}

template <class T, int Threads>
__global__ void fused(const T* x, const T* residual, const T* weight,
                      T* out, int width, float eps) {
  const int64_t base = int64_t(blockIdx.x) * width;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < width; col += Threads) {
    const float z = read_value(x[base + col]) + read_value(residual[base + col]);
    sum += z * z;
  }
  const float scale = rsqrtf(row_sum<Threads>(sum) / float(width) + eps);
  // Reloading avoids a width-dependent register array and supports arbitrary tails.
  for (int col = threadIdx.x; col < width; col += Threads) {
    const float z = read_value(x[base + col]) + read_value(residual[base + col]);
    out[base + col] = write_value<T>((z * scale) * read_value(weight[col]));
  }
}

template <class T>
__global__ void residual_add(const T* x, const T* residual, float* scratch, int64_t n) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) scratch[i] = read_value(x[i]) + read_value(residual[i]);
}

template <class T>
__global__ void normalize(const float* scratch, const T* weight, T* out,
                          int width, float eps) {
  const int64_t base = int64_t(blockIdx.x) * width;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < width; col += 256) {
    const float z = scratch[base + col];
    sum += z * z;
  }
  const float scale = rsqrtf(row_sum<256>(sum) / float(width) + eps);
  for (int col = threadIdx.x; col < width; col += 256)
    out[base + col] = write_value<T>((scratch[base + col] * scale) * read_value(weight[col]));
}

template <class T>
cudaError_t dispatch(const void* x, const void* residual, const void* weight,
                     void* out, float* scratch, int rows, int width,
                     float eps, int variant, cudaStream_t stream) {
  const T* a = static_cast<const T*>(x);
  const T* b = static_cast<const T*>(residual);
  const T* w = static_cast<const T*>(weight);
  T* y = static_cast<T*>(out);
  if (variant == 2) {
    const int64_t n = int64_t(rows) * width;
    residual_add<T><<<unsigned((n + 255) / 256), 256, 0, stream>>>(a, b, scratch, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    normalize<T><<<rows, 256, 0, stream>>>(scratch, w, y, width, eps);
  } else if (variant == 1 && width <= 128) {
    fused<T, 32><<<rows, 32, 0, stream>>>(a, b, w, y, width, eps);
  } else if (variant == 1 && width <= 1024) {
    fused<T, 128><<<rows, 128, 0, stream>>>(a, b, w, y, width, eps);
  } else {
    fused<T, 256><<<rows, 256, 0, stream>>>(a, b, w, y, width, eps);
  }
  return cudaGetLastError();
}

// C ABI accepts contiguous, nonoverlapping device buffers on the caller's stream.
// dtype: 0=FP32, 1=FP16. variant: 0=fused256, 1=width dispatch, 2=two launches.
extern "C" int rmsnorm(const void* x, const void* residual, const void* weight,
                       void* out, float* scratch, int rows, int width,
                       float eps, int dtype, int variant, void* stream) {
  if (rows < 0 || width <= 0 || width > 1048576 || !std::isfinite(eps) || eps <= 0 ||
      dtype < 0 || dtype > 1 || variant < 0 || variant > 2)
    return int(cudaErrorInvalidValue);
  if (rows == 0) return int(cudaSuccess);
  if (!x || !residual || !weight || !out || (variant == 2 && !scratch))
    return int(cudaErrorInvalidValue);
  const int64_t n = int64_t(rows) * width;
  if (variant == 2 && (n + 255) / 256 > 2147483647LL)
    return int(cudaErrorInvalidConfiguration);
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  return int(dtype == 0 ? dispatch<float>(x, residual, weight, out, scratch, rows, width, eps, variant, s)
                       : dispatch<half>(x, residual, weight, out, scratch, rows, width, eps, variant, s));
}

extern "C" const char* rmsnorm_error(int error) { return cudaGetErrorString(cudaError_t(error)); }
