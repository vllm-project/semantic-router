# Source-built native dependency and Router binary for Linux/RISC-V64.
# Build context: repository root. Native builds supply a verified SDK image.
ARG IMAGE_REGISTRY=docker.io/
ARG GO_IMAGE=library/golang:1.25-bookworm
ARG RUST_IMAGE=library/rust:1.90-slim-bookworm

FROM --platform=$BUILDPLATFORM ${IMAGE_REGISTRY}${RUST_IMAGE} AS glide-build
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential protobuf-compiler cmake pkg-config curl ca-certificates libssl-dev \
    && if [ "$(dpkg --print-architecture)" != riscv64 ]; then \
      apt-get install -y --no-install-recommends gcc-riscv64-linux-gnu g++-riscv64-linux-gnu; \
    fi && rm -rf /var/lib/apt/lists/*
RUN if [ ! -d "$(rustc --print target-libdir --target riscv64gc-unknown-linux-gnu)" ]; then \
      rustup target add riscv64gc-unknown-linux-gnu; \
    fi
WORKDIR /source
COPY src/semantic-router/go.mod src/semantic-router/go.mod
COPY tools/docker/riscv64/build-glide.sh tools/docker/riscv64/build-glide.sh
COPY tools/docker/riscv64/valkey-glide/ tools/docker/riscv64/valkey-glide/
RUN bash tools/docker/riscv64/build-glide.sh --output /out/glide

FROM --platform=$BUILDPLATFORM ${IMAGE_REGISTRY}${GO_IMAGE} AS router-build
ARG TARGETARCH
RUN test "$TARGETARCH" = riscv64 && \
    if [ "$(dpkg --print-architecture)" != riscv64 ]; then \
      apt-get update && apt-get install -y --no-install-recommends gcc-riscv64-linux-gnu libc6-dev-riscv64-cross && \
      rm -rf /var/lib/apt/lists/*; \
    fi
WORKDIR /src/semantic-router
ARG GOPROXY="https://proxy.golang.org|direct"
ENV GOPROXY=${GOPROXY}
COPY src/semantic-router/go.mod src/semantic-router/go.sum ./
RUN go mod download
COPY src/semantic-router/ ./
COPY --from=glide-build /out/glide/ /out/glide/
RUN if [ "$(dpkg --print-architecture)" != riscv64 ]; then \
      export CC=riscv64-linux-gnu-gcc; \
    fi && \
    CGO_ENABLED=1 GOOS=linux GOARCH=riscv64 \
      go build -trimpath -buildvcs=false -ldflags="-s -w -extldflags=-L/out/glide" \
      -o /out/router ./cmd

FROM scratch AS router-binary
COPY --from=router-build /out/router /out/router
