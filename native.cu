#include "kernels.cu"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define CHECK(call) do { auto e = (call); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); std::exit(2); } } while (0)

template <class T> float host_float(T x) { return float(x); }
template <> float host_float<half>(half x) { return __half2float(x); }
template <class T> T host_value(float x) { return T(x); }
template <> half host_value<half>(float x) { return __float2half_rn(x); }

template <class T> void run_case(int rows, int width, bool benchmark, int only_variant = -1) {
  const size_t n = size_t(rows) * width;
  std::vector<T> x(n), r(n), w(width), actual(n);
  for (size_t i = 0; i < n; ++i) {
    x[i] = host_value<T>((int(i % 37) - 18) / 16.0f);
    r[i] = host_value<T>((int(i % 19) - 9) / 32.0f);
  }
  for (int i = 0; i < width; ++i) w[i] = host_value<T>((int(i % 13) - 6) / 8.0f);
  T *dx, *dr, *dw, *dy;
  float* scratch;
  CHECK(cudaMalloc(&dx, n * sizeof(T))); CHECK(cudaMalloc(&dr, n * sizeof(T)));
  CHECK(cudaMalloc(&dw, width * sizeof(T))); CHECK(cudaMalloc(&dy, n * sizeof(T)));
  CHECK(cudaMalloc(&scratch, n * sizeof(float)));
  CHECK(cudaMemcpy(dx, x.data(), n * sizeof(T), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dr, r.data(), n * sizeof(T), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dw, w.data(), width * sizeof(T), cudaMemcpyHostToDevice));
  const int dtype = sizeof(T) == 2;
  for (int variant = 0; variant < 3; ++variant) {
    if (only_variant >= 0 && only_variant != variant) continue;
    CHECK(cudaMemset(dy, 0xff, n * sizeof(T)));
    CHECK(cudaError_t(rmsnorm(dx, dr, dw, dy, scratch, rows, width, 1e-5f, dtype, variant, nullptr)));
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(actual.data(), dy, n * sizeof(T), cudaMemcpyDeviceToHost));
    double max_error = 0;
    for (int row = 0; row < rows; ++row) {
      double sum = 0;
      for (int col = 0; col < width; ++col) {
        size_t i = size_t(row) * width + col;
        double z = double(host_float(x[i])) + host_float(r[i]); sum += z * z;
      }
      double scale = 1.0 / std::sqrt(sum / width + double(1e-5f));
      for (int col = 0; col < width; ++col) {
        size_t i = size_t(row) * width + col;
        double z = double(host_float(x[i])) + host_float(r[i]);
        double expected = host_float(host_value<T>(float(z * scale * host_float(w[col]))));
        double error = std::abs(double(host_float(actual[i])) - expected);
        max_error = std::max(max_error, error);
        double tolerance = dtype ? 0.002 + 0.002 * std::abs(expected) : 2e-6 + 2e-5 * std::abs(expected);
        if (!std::isfinite(host_float(actual[i])) || error > tolerance) {
          std::fprintf(stderr, "FAIL dtype=%d rows=%d width=%d variant=%d index=%zu\n", dtype, rows, width, variant, i);
          std::exit(1);
        }
      }
    }
    if (!benchmark) std::printf("PASS dtype=%d rows=%d width=%d variant=%d max_abs=%g\n", dtype, rows, width, variant, max_error);
    if (benchmark) {
      cudaEvent_t start, stop; CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&stop));
      for (int i = 0; i < 30; ++i) CHECK(cudaError_t(rmsnorm(dx, dr, dw, dy, scratch, rows, width, 1e-5f, dtype, variant, nullptr)));
      CHECK(cudaDeviceSynchronize());
      std::vector<float> samples;
      for (int group = 0; group < 7; ++group) {
        CHECK(cudaEventRecord(start));
        for (int i = 0; i < 200; ++i) CHECK(cudaError_t(rmsnorm(dx, dr, dw, dy, scratch, rows, width, 1e-5f, dtype, variant, nullptr)));
        CHECK(cudaEventRecord(stop)); CHECK(cudaEventSynchronize(stop));
        float ms; CHECK(cudaEventElapsedTime(&ms, start, stop)); samples.push_back(ms * 1000 / 200);
      }
      std::printf("TIMING dtype=%d rows=%d width=%d variant=%d samples_us=", dtype, rows, width, variant);
      for (float s : samples) std::printf("%.6f,", s);
      std::sort(samples.begin(), samples.end());
      std::printf(" median_us=%.6f min_us=%.6f max_us=%.6f\n", samples[3], samples.front(), samples.back());
      CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(stop));
    }
  }
  CHECK(cudaFree(dx)); CHECK(cudaFree(dr)); CHECK(cudaFree(dw)); CHECK(cudaFree(dy)); CHECK(cudaFree(scratch));
}

int main(int argc, char** argv) {
  // A single explicit workload keeps profiler replay separate from timing loops.
  if (argc == 6) {
    int rows = std::atoi(argv[2]), width = std::atoi(argv[3]);
    int dtype = std::atoi(argv[4]), variant = std::atoi(argv[5]);
    if (rows <= 0 || rows > 4096 || width <= 0 || width > 8192 ||
        dtype < 0 || dtype > 1 || variant < 0 || variant > 2) return 2;
    if (dtype) run_case<half>(rows, width, false, variant);
    else run_case<float>(rows, width, false, variant);
    return 0;
  }
  bool benchmark = argc > 1;
  if (!benchmark) {
    for (int width : {1, 7, 31, 32, 33, 127, 128, 129, 255, 256, 257, 1023, 1024, 1025, 4096, 4099, 8192}) {
      run_case<float>(3, width, false); run_case<half>(3, width, false);
    }
    std::puts("PASS native_checks=102");
  } else {
    const int shapes[][2] = {{1,128},{1,4096},{32,768},{128,1024},{1024,4096},{1024,4099},{256,8192}};
    for (auto& s : shapes) { run_case<float>(s[0], s[1], true); run_case<half>(s[0], s[1], true); }
  }
}
