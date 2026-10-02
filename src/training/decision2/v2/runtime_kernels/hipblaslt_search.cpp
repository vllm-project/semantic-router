// Per-shape hipBLASLt algorithm search for BF16 Linear layers (y = x W^T), the counterpart of
// open-jev-fast's cuBLASLt timing (written from the hipBLASLt API docs; no code copied).
//
//   hipcc -O2 -std=c++17 hipblaslt_search.cpp -lhipblaslt -o hipblaslt_search
//   ./hipblaslt_search shapes.txt [quick_iters=3] [top=16] [iters=50] > results.jsonl
//
// shapes.txt: one "M N K" per line. The GEMM is laid out exactly as PyTorch's F.linear calls it
// (column-major m=N, n=M, k=K, opA=T on W [N, K], opB=N on x [M, K], BF16 in/out, FP32 compute).
// For every shape it times the heuristic's first choice (what PyTorch's hipBLASLt path runs) and
// every algorithm hipblaslt_ext::getAllAlgos returns that supports the problem: a quick pass of
// quick_iters calls each, then the best `top` again with `iters` calls; for those it also tries
// split-K (GemmTuning.splitK = 2, 4, 8, 16). Weights rotate over enough copies to exceed the
// 256 MB Infinity Cache, as a real forward (new weights every layer) never hits it.
// Output: one JSON object per shape.
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#define CHECK_HIP(x)                                                                    \
  do {                                                                                  \
    hipError_t e = (x);                                                                 \
    if (e != hipSuccess) {                                                              \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
      exit(1);                                                                          \
    }                                                                                   \
  } while (0)
#define CHECK_BLAS(x)                                                         \
  do {                                                                        \
    hipblasStatus_t s = (x);                                                  \
    if (s != HIPBLAS_STATUS_SUCCESS) {                                        \
      fprintf(stderr, "hipBLASLt error %d at %s:%d\n", (int)s, __FILE__, __LINE__); \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

static const size_t kWorkspace = 128ull << 20;
static const size_t kRotateBytes = 640ull << 20;

struct Problem {
  int64_t M, N, K;
  std::vector<void*> weights;
  void* x = nullptr;
  void* y = nullptr;
  hipblasLtMatmulDesc_t desc;
  hipblasLtMatrixLayout_t a, b, c;
};

struct Timer {
  hipEvent_t start, stop;
  Timer() {
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));
  }
};

static void fill_random(void* ptr, size_t elems, unsigned seed) {
  std::vector<uint16_t> host(elems);
  uint32_t s = seed * 2654435761u + 1;
  for (size_t i = 0; i < elems; ++i) {
    s = s * 1664525u + 1013904223u;
    float f = ((s >> 8) & 0xffff) / 65536.0f - 0.5f;
    uint32_t bits;
    memcpy(&bits, &f, 4);
    host[i] = (uint16_t)(bits >> 16);
  }
  CHECK_HIP(hipMemcpy(ptr, host.data(), elems * 2, hipMemcpyHostToDevice));
}

// Mean microseconds per call over `iters` back-to-back calls (weights rotate), -1 on failure.
template <typename Launch>
static double time_calls(Launch launch, int copies, int iters, hipStream_t stream, Timer& t) {
  for (int i = 0; i < 2; ++i)
    if (!launch(i % copies)) return -1;
  CHECK_HIP(hipStreamSynchronize(stream));
  CHECK_HIP(hipEventRecord(t.start, stream));
  for (int i = 0; i < iters; ++i)
    if (!launch(i % copies)) return -1;
  CHECK_HIP(hipEventRecord(t.stop, stream));
  CHECK_HIP(hipEventSynchronize(t.stop));
  float ms = 0;
  CHECK_HIP(hipEventElapsedTime(&ms, t.start, t.stop));
  return 1000.0 * ms / iters;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s shapes.txt [quick_iters] [top] [iters]\n", argv[0]);
    return 2;
  }
  int quick = argc > 2 ? atoi(argv[2]) : 3;
  int top = argc > 3 ? atoi(argv[3]) : 16;
  int iters = argc > 4 ? atoi(argv[4]) : 50;
  std::vector<std::array<int64_t, 3>> shapes;
  {
    std::ifstream in(argv[1]);
    int64_t m, n, k;
    while (in >> m >> n >> k) shapes.push_back({m, n, k});
  }
  hipblasLtHandle_t handle;
  CHECK_BLAS(hipblasLtCreate(&handle));
  hipStream_t stream;
  CHECK_HIP(hipStreamCreate(&stream));
  void* workspace;
  CHECK_HIP(hipMalloc(&workspace, kWorkspace));
  Timer timer;
  const float alpha = 1.0f, beta = 0.0f;
  hipblasOperation_t opA = HIPBLAS_OP_T, opB = HIPBLAS_OP_N;

  std::vector<hipblasLtMatmulHeuristicResult_t> all;
  CHECK_BLAS(hipblaslt_ext::getAllAlgos(handle, hipblaslt_ext::GemmType::HIPBLASLT_GEMM, opA, opB, HIP_R_16BF,
                                        HIP_R_16BF, HIP_R_16BF, HIP_R_16BF, HIPBLAS_COMPUTE_32F, all));
  fprintf(stderr, "getAllAlgos: %zu algorithms\n", all.size());

  for (auto& s : shapes) {
    Problem p;
    p.M = s[0];
    p.N = s[1];
    p.K = s[2];
    size_t wbytes = (size_t)p.N * p.K * 2;
    int copies = (int)std::min<size_t>(64, std::max<size_t>(1, (kRotateBytes + wbytes - 1) / wbytes));
    for (int i = 0; i < copies; ++i) {
      void* w;
      CHECK_HIP(hipMalloc(&w, wbytes));
      fill_random(w, (size_t)p.N * p.K, 17 + i);
      p.weights.push_back(w);
    }
    CHECK_HIP(hipMalloc(&p.x, (size_t)p.M * p.K * 2));
    fill_random(p.x, (size_t)p.M * p.K, 3);
    CHECK_HIP(hipMalloc(&p.y, (size_t)p.M * p.N * 2));
    CHECK_BLAS(hipblasLtMatmulDescCreate(&p.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    CHECK_BLAS(hipblasLtMatmulDescSetAttribute(p.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    CHECK_BLAS(hipblasLtMatmulDescSetAttribute(p.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));
    CHECK_BLAS(hipblasLtMatrixLayoutCreate(&p.a, HIP_R_16BF, p.K, p.N, p.K));
    CHECK_BLAS(hipblasLtMatrixLayoutCreate(&p.b, HIP_R_16BF, p.K, p.M, p.K));
    CHECK_BLAS(hipblasLtMatrixLayoutCreate(&p.c, HIP_R_16BF, p.N, p.M, p.N));

    auto run_algo = [&](hipblasLtMatmulAlgo_t* algo) {
      return [&, algo](int i) {
        return hipblasLtMatmul(handle, p.desc, &alpha, p.weights[i], p.a, p.x, p.b, &beta, p.y, p.c, p.y, p.c, algo,
                               workspace, kWorkspace, stream) == HIPBLAS_STATUS_SUCCESS;
      };
    };

    // heuristic first choice (PyTorch's hipBLASLt path asks for one result)
    hipblasLtMatmulPreference_t pref;
    CHECK_BLAS(hipblasLtMatmulPreferenceCreate(&pref));
    CHECK_BLAS(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &kWorkspace,
                                                    sizeof(kWorkspace)));
    hipblasLtMatmulHeuristicResult_t heur[1];
    int returned = 0;
    hipblasLtMatmulAlgoGetHeuristic(handle, p.desc, p.a, p.b, p.c, p.c, pref, 1, heur, &returned);
    double heuristic_us = -1;
    int heuristic_index = -1;
    if (returned > 0) {
      heuristic_us = time_calls(run_algo(&heur[0].algo), copies, iters, stream, timer);
      heuristic_index = hipblaslt_ext::getIndexFromAlgo(heur[0].algo);
    }

    // every supported algorithm, quick pass
    std::vector<std::pair<double, int>> quick_times;
    int supported = 0;
    for (size_t i = 0; i < all.size(); ++i) {
      size_t ws = 0;
      if (hipblaslt_ext::matmulIsAlgoSupported(handle, p.desc, &alpha, p.a, p.b, &beta, p.c, p.c, all[i].algo, ws) !=
              HIPBLAS_STATUS_SUCCESS ||
          ws > kWorkspace)
        continue;
      ++supported;
      double us = time_calls(run_algo(&all[i].algo), copies, quick, stream, timer);
      if (us > 0) quick_times.push_back({us, (int)i});
    }
    std::sort(quick_times.begin(), quick_times.end());
    std::vector<std::pair<double, int>> refined;
    for (int j = 0; j < (int)quick_times.size() && j < top; ++j) {
      int i = quick_times[j].second;
      double us = time_calls(run_algo(&all[i].algo), copies, iters, stream, timer);
      if (us > 0) refined.push_back({us, i});
    }
    std::sort(refined.begin(), refined.end());

    // split-K on the refined candidates through the extension API
    double best_splitk_us = -1;
    int best_splitk = 0, best_splitk_index = -1;
#ifdef SPLITK
    hipblaslt_ext::Gemm gemm(handle, opA, opB, HIP_R_16BF, HIP_R_16BF, HIP_R_16BF, HIP_R_16BF, HIPBLAS_COMPUTE_32F);
    hipblaslt_ext::GemmEpilogue epilogue;
    hipblaslt_ext::GemmInputs inputs;
    inputs.setA(p.weights[0]);
    inputs.setB(p.x);
    inputs.setC(p.y);
    inputs.setD(p.y);
    inputs.setAlpha(&alpha);
    inputs.setBeta(&beta);
    gemm.setProblem(p.N, p.M, p.K, 1, epilogue, inputs);
    for (int j = 0; j < (int)refined.size() && j < 8; ++j) {
      int i = refined[j].second;
      for (int sk : {2, 4, 8, 16}) {
        hipblaslt_ext::GemmTuning tuning;
        tuning.splitK = sk;
        size_t ws = 0;
        if (gemm.isAlgoSupported(all[i].algo, tuning, ws) != HIPBLAS_STATUS_SUCCESS || ws > kWorkspace) continue;
        if (gemm.initialize(all[i].algo, tuning, workspace) != HIPBLAS_STATUS_SUCCESS) continue;
        // weights are bound at initialize; rotating would need re-initialisation, so time copy 0 only
        auto launch0 = [&](int) { return gemm.run(stream) == HIPBLAS_STATUS_SUCCESS; };
        double us = time_calls(launch0, 1, iters, stream, timer);
        if (us > 0 && (best_splitk_us < 0 || us < best_splitk_us)) {
          best_splitk_us = us;
          best_splitk = sk;
          best_splitk_index = hipblaslt_ext::getIndexFromAlgo(all[i].algo);
        }
      }
    }
#endif
    // the same refined best without split-K, also timed on copy 0 only (comparable with split-K)
    double best_copy0_us = -1;
    if (!refined.empty()) best_copy0_us = time_calls(run_algo(&all[refined[0].second].algo), 1, iters, stream, timer);

    printf("{\"M\": %ld, \"N\": %ld, \"K\": %ld, \"weight_copies\": %d, \"supported\": %d, "
           "\"heuristic_us\": %.3f, \"heuristic_index\": %d, \"best_us\": %.3f, \"best_index\": %d, "
           "\"best_copy0_us\": %.3f, \"best_splitk_us\": %.3f, \"best_splitk\": %d, \"best_splitk_index\": %d, "
           "\"top\": [",
           (long)p.M, (long)p.N, (long)p.K, copies, supported, heuristic_us, heuristic_index,
           refined.empty() ? -1.0 : refined[0].first,
           refined.empty() ? -1 : hipblaslt_ext::getIndexFromAlgo(all[refined[0].second].algo), best_copy0_us,
           best_splitk_us, best_splitk, best_splitk_index);
    for (int j = 0; j < (int)refined.size() && j < 5; ++j)
      printf("%s[%.3f, %d]", j ? ", " : "", refined[j].first, hipblaslt_ext::getIndexFromAlgo(all[refined[j].second].algo));
    printf("]}\n");
    fflush(stdout);

    hipblasLtMatmulPreferenceDestroy(pref);
    hipblasLtMatrixLayoutDestroy(p.a);
    hipblasLtMatrixLayoutDestroy(p.b);
    hipblasLtMatrixLayoutDestroy(p.c);
    hipblasLtMatmulDescDestroy(p.desc);
    for (auto w : p.weights) CHECK_HIP(hipFree(w));
    CHECK_HIP(hipFree(p.x));
    CHECK_HIP(hipFree(p.y));
  }
  return 0;
}
