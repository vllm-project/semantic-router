"""ROCm (MI325X, gfx942) kernel studies for the Decision 2.0 runtime: research code, not the release path.

Kernel-level counterparts of open-jev-fast's phase 2-3 CUDA work (MIT; used as a
reference only, no code copied): fused element-wise Triton kernels, Gated
DeltaNet chunk-kernel tuning, tree attention and hipBLASLt per-shape GEMM
timing, each with a benchmark against the shipped runtime's own op sequence.
Nothing here is imported by the released packages; adoption goes through the
parity rollout.
"""
