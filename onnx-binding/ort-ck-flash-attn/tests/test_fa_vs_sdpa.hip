// Validation test: compare CK Flash Attention output against a reference
// FP32 SDPA (Scaled Dot-Product Attention) computed on CPU.
//
// Reference computes:  O = softmax(Q @ K^T * scale [+ bias]) @ V  in FP32.
// CK FA computes the same in FP16 on GPU via tiled flash attention.
// We compare element-wise and report max/mean absolute error and cosine similarity.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#include "ck_flash_attn.h"

using half_t = _Float16;

#define HIP_CHECK(call)                                                       \
    do {                                                                      \
        hipError_t err = (call);                                              \
        if (err != hipSuccess) {                                              \
            fprintf(stderr, "HIP error %d at %s:%d: %s\n", err, __FILE__,    \
                    __LINE__, hipGetErrorString(err));                         \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// ── FP16 <-> FP32 helpers ──────────────────────────────────────
static float to_f32(half_t h) { return static_cast<float>(h); }
static half_t to_f16(float f) { return static_cast<half_t>(f); }

// ── Reference SDPA in FP32 on CPU ──────────────────────────────
// Q, K, V are in [B, H, S, D] layout, stored as FP16.
// bias (optional) is [B, H, Sq, Sk] in FP16.
// Returns O in FP32 [B, H, Sq, D].
static std::vector<float> reference_sdpa_f32(
    const std::vector<half_t>& q_f16,
    const std::vector<half_t>& k_f16,
    const std::vector<half_t>& v_f16,
    const std::vector<half_t>& bias_f16,
    int batch, int nhead, int seqlen_q, int seqlen_k, int hdim, float scale)
{
    // Upcast inputs to FP32
    auto upcast = [](const std::vector<half_t>& src) {
        std::vector<float> dst(src.size());
        for (size_t i = 0; i < src.size(); i++)
            dst[i] = to_f32(src[i]);
        return dst;
    };

    std::vector<float> q = upcast(q_f16);
    std::vector<float> k = upcast(k_f16);
    std::vector<float> v = upcast(v_f16);
    std::vector<float> bias;
    bool has_bias = !bias_f16.empty();
    if (has_bias) bias = upcast(bias_f16);

    size_t o_size = (size_t)batch * nhead * seqlen_q * hdim;
    std::vector<float> out(o_size, 0.0f);

    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < nhead; h++) {
            // Base offsets for this (b, h) slice
            size_t qo_base = ((size_t)b * nhead + h) * seqlen_q * hdim;
            size_t k_base  = ((size_t)b * nhead + h) * seqlen_k * hdim;
            size_t v_base  = ((size_t)b * nhead + h) * seqlen_k * hdim;
            size_t bias_base = has_bias
                ? ((size_t)b * nhead + h) * seqlen_q * seqlen_k
                : 0;

            // For each query position
            for (int sq = 0; sq < seqlen_q; sq++) {
                // Compute scores: S[sk] = sum_d(Q[sq,d] * K[sk,d]) * scale
                std::vector<float> scores(seqlen_k);
                for (int sk = 0; sk < seqlen_k; sk++) {
                    float dot = 0.0f;
                    for (int d = 0; d < hdim; d++) {
                        dot += q[qo_base + sq * hdim + d]
                             * k[k_base + sk * hdim + d];
                    }
                    scores[sk] = dot * scale;
                    if (has_bias) {
                        scores[sk] += bias[bias_base + sq * seqlen_k + sk];
                    }
                }

                // Softmax (numerically stable)
                float max_s = *std::max_element(scores.begin(), scores.end());
                float sum_exp = 0.0f;
                for (int sk = 0; sk < seqlen_k; sk++) {
                    scores[sk] = expf(scores[sk] - max_s);
                    sum_exp += scores[sk];
                }
                for (int sk = 0; sk < seqlen_k; sk++) {
                    scores[sk] /= sum_exp;
                }

                // Output: O[sq, d] = sum_sk(scores[sk] * V[sk, d])
                for (int d = 0; d < hdim; d++) {
                    float val = 0.0f;
                    for (int sk = 0; sk < seqlen_k; sk++) {
                        val += scores[sk] * v[v_base + sk * hdim + d];
                    }
                    out[qo_base + sq * hdim + d] = val;
                }
            }
        }
    }

    return out;
}

// ── Test metrics ───────────────────────────────────────────────
struct TestMetrics {
    float max_abs_err;
    float mean_abs_err;
    float cosine_sim;
    float rmse;
    size_t n_elements;
};

static TestMetrics compute_metrics(const std::vector<float>& ref,
                                   const std::vector<float>& test) {
    assert(ref.size() == test.size());
    size_t n = ref.size();

    float max_ae = 0.0f;
    double sum_ae = 0.0;
    double sum_se = 0.0;
    double dot_rt = 0.0, norm_r = 0.0, norm_t = 0.0;

    for (size_t i = 0; i < n; i++) {
        float ae = fabsf(ref[i] - test[i]);
        max_ae = std::max(max_ae, ae);
        sum_ae += ae;
        sum_se += (double)ae * ae;
        dot_rt += (double)ref[i] * test[i];
        norm_r += (double)ref[i] * ref[i];
        norm_t += (double)test[i] * test[i];
    }

    TestMetrics m{};
    m.n_elements = n;
    m.max_abs_err = max_ae;
    m.mean_abs_err = (float)(sum_ae / n);
    m.rmse = (float)sqrt(sum_se / n);
    m.cosine_sim = (norm_r > 0 && norm_t > 0)
        ? (float)(dot_rt / (sqrt(norm_r) * sqrt(norm_t)))
        : 0.0f;
    return m;
}

// ── Single test case ───────────────────────────────────────────
struct TestConfig {
    int batch;
    int nhead;
    int seqlen_q;
    int seqlen_k;
    int hdim;
    bool use_bias;
    float max_abs_tol;
    float min_cosine_tol;
};

static bool run_test(const TestConfig& cfg) {
    const float scale = 1.0f / sqrtf((float)cfg.hdim);
    const size_t n_qo = (size_t)cfg.batch * cfg.nhead * cfg.seqlen_q * cfg.hdim;
    const size_t n_kv = (size_t)cfg.batch * cfg.nhead * cfg.seqlen_k * cfg.hdim;
    const size_t n_bias = cfg.use_bias
        ? (size_t)cfg.batch * cfg.nhead * cfg.seqlen_q * cfg.seqlen_k
        : 0;

    // Generate random FP16 inputs in [-0.5, 0.5]
    auto rand_fp16 = [](size_t n, unsigned seed) {
        std::vector<half_t> buf(n);
        srand(seed);
        for (size_t i = 0; i < n; i++)
            buf[i] = to_f16(((float)(rand() % 1000) / 1000.0f) - 0.5f);
        return buf;
    };

    std::vector<half_t> h_q = rand_fp16(n_qo, 42);
    std::vector<half_t> h_k = rand_fp16(n_kv, 137);
    std::vector<half_t> h_v = rand_fp16(n_kv, 7);

    std::vector<half_t> h_bias;
    if (cfg.use_bias) {
        h_bias.resize(n_bias);
        srand(999);
        for (size_t i = 0; i < n_bias; i++) {
            // Simulate a padding mask: ~10% of positions masked with -1e4
            float val = (rand() % 10 == 0) ? -1e4f : 0.0f;
            h_bias[i] = to_f16(val);
        }
    }

    // ── Reference SDPA (CPU, FP32) ──
    std::vector<float> ref_out = reference_sdpa_f32(
        h_q, h_k, h_v, h_bias,
        cfg.batch, cfg.nhead, cfg.seqlen_q, cfg.seqlen_k, cfg.hdim, scale);

    // ── CK Flash Attention (GPU, FP16) ──
    void *d_q, *d_k, *d_v, *d_o;
    void *d_bias = nullptr;
    HIP_CHECK(hipMalloc(&d_q, n_qo * sizeof(half_t)));
    HIP_CHECK(hipMalloc(&d_k, n_kv * sizeof(half_t)));
    HIP_CHECK(hipMalloc(&d_v, n_kv * sizeof(half_t)));
    HIP_CHECK(hipMalloc(&d_o, n_qo * sizeof(half_t)));
    if (cfg.use_bias) {
        HIP_CHECK(hipMalloc(&d_bias, n_bias * sizeof(half_t)));
        HIP_CHECK(hipMemcpy(d_bias, h_bias.data(), n_bias * sizeof(half_t),
                             hipMemcpyHostToDevice));
    }

    HIP_CHECK(hipMemcpy(d_q, h_q.data(), n_qo * sizeof(half_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_k, h_k.data(), n_kv * sizeof(half_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_v, h_v.data(), n_kv * sizeof(half_t), hipMemcpyHostToDevice));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    int32_t nhead_bias = cfg.use_bias ? cfg.nhead : 0;
    int rc = ck_flash_attn_fwd(
        stream, d_q, d_k, d_v, d_bias, d_o,
        cfg.batch, cfg.nhead, cfg.seqlen_q, cfg.seqlen_k, cfg.hdim, nhead_bias, scale);
    HIP_CHECK(hipStreamSynchronize(stream));

    if (rc != 0) {
        fprintf(stderr, "  ck_flash_attn_fwd returned %d\n", rc);
        HIP_CHECK(hipFree(d_q)); HIP_CHECK(hipFree(d_k));
        HIP_CHECK(hipFree(d_v)); HIP_CHECK(hipFree(d_o));
        if (d_bias) HIP_CHECK(hipFree(d_bias));
        HIP_CHECK(hipStreamDestroy(stream));
        return false;
    }

    std::vector<half_t> h_o(n_qo);
    HIP_CHECK(hipMemcpy(h_o.data(), d_o, n_qo * sizeof(half_t), hipMemcpyDeviceToHost));

    // Upcast FA output to FP32 for comparison
    std::vector<float> fa_out(n_qo);
    for (size_t i = 0; i < n_qo; i++)
        fa_out[i] = to_f32(h_o[i]);

    // Compute metrics
    TestMetrics m = compute_metrics(ref_out, fa_out);

    bool pass = (m.max_abs_err <= cfg.max_abs_tol)
             && (m.cosine_sim >= cfg.min_cosine_tol)
             && (rc == 0);

    printf("  %s  B=%d H=%d Sq=%4d Sk=%4d D=%d bias=%d  "
           "max_ae=%.4e mean_ae=%.4e rmse=%.4e cos=%.8f\n",
           pass ? "PASS" : "FAIL",
           cfg.batch, cfg.nhead, cfg.seqlen_q, cfg.seqlen_k,
           cfg.hdim, cfg.use_bias,
           m.max_abs_err, m.mean_abs_err, m.rmse, m.cosine_sim);

    if (!pass) {
        fprintf(stderr, "    FAILED: max_abs_err=%.4e (tol %.4e), cosine=%.8f (tol %.4f)\n",
                m.max_abs_err, cfg.max_abs_tol, m.cosine_sim, cfg.min_cosine_tol);
    }

    HIP_CHECK(hipFree(d_q)); HIP_CHECK(hipFree(d_k));
    HIP_CHECK(hipFree(d_v)); HIP_CHECK(hipFree(d_o));
    if (d_bias) HIP_CHECK(hipFree(d_bias));
    HIP_CHECK(hipStreamDestroy(stream));
    return pass;
}

// ── Main test suite ────────────────────────────────────────────
int main() {
    printf("=== CK Flash Attention vs Reference SDPA Validation ===\n");
    printf("  Reference: FP32 dense attention on CPU\n");
    printf("  Test:      FP16 CK Flash Attention on GPU (MI300X)\n\n");

    // Tolerance rationale:
    // - FP16 has ~3.3 decimal digits of precision (eps ~= 9.77e-4)
    // - Softmax amplifies errors through exp() and normalization
    // - For moderate seq lengths, max absolute error typically < 0.05
    // - For longer sequences, softmax becomes more peaked, absolute errors can grow
    //   but cosine similarity should remain very high (> 0.999)
    // - With bias (mask), masked positions get near-zero attention weight,
    //   which is well-behaved numerically.

    std::vector<TestConfig> tests = {
        // Basic correctness: hdim=64, the mmBERT config
        {1, 1,   64,   64, 64, false, 0.05f, 0.9995f},
        {1, 1,  128,  128, 64, false, 0.05f, 0.9995f},
        {1, 1,  256,  256, 64, false, 0.05f, 0.9990f},
        {1, 1,  512,  512, 64, false, 0.05f, 0.9990f},

        // Multi-head (12 heads like mmBERT)
        {1, 12,  64,   64, 64, false, 0.05f, 0.9995f},
        {1, 12, 128,  128, 64, false, 0.05f, 0.9995f},
        {1, 12, 256,  256, 64, false, 0.05f, 0.9990f},

        // Batch > 1
        {2, 12,  128, 128, 64, false, 0.05f, 0.9995f},
        {4, 12,   64,  64, 64, false, 0.05f, 0.9995f},

        // With additive attention bias (padding mask)
        {1, 1,  128, 128, 64, true, 0.05f, 0.9995f},
        {1, 12, 128, 128, 64, true, 0.05f, 0.9995f},
        {1, 12, 256, 256, 64, true, 0.05f, 0.9990f},
        {2, 12, 128, 128, 64, true, 0.05f, 0.9995f},

        // Other head dims
        {1, 1, 128, 128, 32, false, 0.05f, 0.9995f},
        {1, 1, 128, 128, 128, false, 0.05f, 0.9995f},

        // Longer sequences (only 1 head to keep CPU reference tractable)
        {1, 1, 1024, 1024, 64, false, 0.10f, 0.9980f},
        {1, 1, 2048, 2048, 64, false, 0.10f, 0.9980f},

        // Cross-attention style (Sq != Sk)
        {1, 12,  64, 128, 64, false, 0.05f, 0.9995f},
        {1, 12, 128,  64, 64, false, 0.05f, 0.9995f},
    };

    int n_pass = 0, n_fail = 0;
    for (const auto& t : tests) {
        if (run_test(t))
            n_pass++;
        else
            n_fail++;
    }

    printf("\n=== Results: %d/%d passed", n_pass, n_pass + n_fail);
    if (n_fail > 0)
        printf(", %d FAILED", n_fail);
    printf(" ===\n");

    return n_fail > 0 ? 1 : 0;
}
