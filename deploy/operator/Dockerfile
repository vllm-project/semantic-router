# Build the operator binary
# Using RHEL 10 UBI Golang builder image
# Build context must be the repository root so local replace directives can reach src/semantic-router.
FROM registry.access.redhat.com/ubi10/go-toolset:latest AS builder

# Target architecture - defaults to amd64
ARG TARGETARCH=amd64

USER root

WORKDIR /workspace

# Copy the module manifests needed by local replace directives first for better caching
RUN mkdir -p deploy/operator src/semantic-router candle-binding ml-binding nlp-binding
COPY deploy/operator/go.mod deploy/operator/go.sum deploy/operator/
COPY src/semantic-router/go.mod src/semantic-router/go.sum src/semantic-router/
COPY candle-binding/go.mod candle-binding/semantic-router.go candle-binding/
COPY ml-binding/go.mod ml-binding/ml_binding.go ml-binding/
COPY nlp-binding/go.mod nlp-binding/nlp_binding.go nlp-binding/nlp_binding_mock.go nlp-binding/

WORKDIR /workspace/deploy/operator

# Cache deps before building and copying source so that we don't need to re-download as much
# and so that source changes don't invalidate our downloaded layer
RUN go mod download

# Copy the go source
COPY deploy/operator/ /workspace/deploy/operator/
COPY src/semantic-router/ /workspace/src/semantic-router/

# Build
RUN CGO_ENABLED=0 GOOS=linux GOARCH=${TARGETARCH} go build -a -o /workspace/manager main.go

# Use RHEL 10 UBI minimal for the final image
FROM registry.access.redhat.com/ubi10/ubi-minimal:latest

# Install necessary runtime dependencies
RUN microdnf install -y shadow-utils && \
    microdnf clean all

WORKDIR /

# Create non-root user
RUN useradd -u 65532 -r -g 0 -m -s /sbin/nologin \
    -c "semantic-router-operator user" semantic-router-operator

# Copy the operator binary from builder
COPY --from=builder /workspace/manager .

# Set ownership and permissions
RUN chown semantic-router-operator:0 /manager && \
    chmod 0755 /manager

USER 65532:0

ENTRYPOINT ["/manager"]
