# Build the operator binary
# Using RHEL 10 UBI Golang builder image
FROM registry.access.redhat.com/ubi10/go-toolset:latest AS builder

# Target architecture - defaults to amd64
ARG TARGETARCH=amd64

USER root

WORKDIR /workspace

# Copy go mod and sum files
COPY go.mod go.mod
COPY go.sum go.sum

# Cache deps before building and copying source so that we don't need to re-download as much
# and so that source changes don't invalidate our downloaded layer
RUN go mod download

# Copy the go source
COPY main.go main.go
COPY api/ api/
COPY controllers/ controllers/

# Build
RUN CGO_ENABLED=0 GOOS=linux GOARCH=${TARGETARCH} go build -a -o manager main.go

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
