FROM rocm/dev-ubuntu-24.04:7.0-complete AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build composablekernel-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build/ort-ck-flash-attn
COPY CMakeLists.txt ./
COPY include/ include/
COPY src/ src/

RUN cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGPU_TARGETS=gfx942 \
        -DCMAKE_INSTALL_PREFIX=/opt/ort-ck-flash-attn && \
    cmake --build build --parallel $(nproc) && \
    cmake --install build

# ── Minimal runtime image ───────────────────────────────────────
FROM rocm/dev-ubuntu-24.04:7.0-complete

COPY --from=builder /opt/ort-ck-flash-attn/lib/libort_ck_flash_attn.so /usr/lib/
COPY scripts/ /opt/ort-ck-flash-attn/scripts/

ENV ORT_CK_FLASH_ATTN_LIB=/usr/lib/libort_ck_flash_attn.so
