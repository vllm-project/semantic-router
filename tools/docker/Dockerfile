FROM quay.io/centos/centos:stream10

RUN dnf -y update && \
    dnf -y install epel-release && \
    dnf -y install \
    gcc \
    gcc-c++ \
    make \
    cmake \
    pkg-config \
    wget \
    tar \
    python3 \
    python3-pip \
    openssl-devel \
    gettext \
    ca-certificates && \
    dnf clean all

# Install HuggingFace CLI for model downloading
RUN pip3 install --no-cache-dir huggingface_hub[cli]

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Envoy
ENV ENVOY_VERSION=1.31.7
RUN ARCH=$(uname -m) && \
    case ${ARCH} in \
        x86_64) ENVOY_ARCH="x86_64" ;; \
        aarch64|arm64) ENVOY_ARCH="aarch64" ;; \
        *) echo "Unsupported architecture: ${ARCH}" && exit 1 ;; \
    esac && \
    curl -OL https://github.com/envoyproxy/envoy/releases/download/v${ENVOY_VERSION}/envoy-${ENVOY_VERSION}-linux-${ENVOY_ARCH} && \
    chmod +x envoy-${ENVOY_VERSION}-linux-${ENVOY_ARCH} && \
    mv envoy-${ENVOY_VERSION}-linux-${ENVOY_ARCH} /usr/local/bin/envoy

# Install Golang
ENV GOLANG_VERSION=1.24.1
RUN ARCH=$(uname -m) && \
    case ${ARCH} in \
        x86_64) GO_ARCH="amd64" ;; \
        aarch64|arm64) GO_ARCH="arm64" ;; \
        *) echo "Unsupported architecture: ${ARCH}" && exit 1 ;; \
    esac && \
    curl -OL https://golang.org/dl/go${GOLANG_VERSION}.linux-${GO_ARCH}.tar.gz && \
    tar -C /usr/local -xzf go${GOLANG_VERSION}.linux-${GO_ARCH}.tar.gz && \
    rm go${GOLANG_VERSION}.linux-${GO_ARCH}.tar.gz
ENV PATH="/usr/local/go/bin:${PATH}"
ENV GOPATH="/go"
ENV PATH="/go/bin:${PATH}"

# Set working directory
WORKDIR /app

# Set environment variables
ENV LD_LIBRARY_PATH=/app/candle-binding/target/release
ENV CGO_ENABLED=1
