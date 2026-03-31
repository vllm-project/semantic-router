# Stage 0: Build DSL compiler WASM assets
FROM golang:1.24-alpine AS wasm-builder
WORKDIR /app/src/semantic-router

ENV GOPROXY=https://proxy.golang.org,direct

COPY src/semantic-router/go.mod src/semantic-router/go.sum ./
COPY src/semantic-router/ ./

# Local replace directives point at sibling bindings that are not needed for the
# WASM build. Stub them under /app so `../../*-binding` resolves correctly.
RUN mkdir -p /app/candle-binding /app/ml-binding /app/nlp-binding && \
    echo 'module github.com/vllm-project/semantic-router/candle-binding' > /app/candle-binding/go.mod && \
    echo 'module github.com/vllm-project/semantic-router/ml-binding' > /app/ml-binding/go.mod && \
    echo 'module github.com/vllm-project/semantic-router/nlp-binding' > /app/nlp-binding/go.mod
RUN go mod download
RUN CGO_ENABLED=0 GOOS=js GOARCH=wasm go build \
      -buildvcs=false \
      -ldflags="-s -w" \
      -o /build/signal-compiler.wasm \
      ./cmd/wasm/ && \
    if [ -f "$(go env GOROOT)/lib/wasm/wasm_exec.js" ]; then \
      cp "$(go env GOROOT)/lib/wasm/wasm_exec.js" /build/wasm_exec.js; \
    else \
      cp "$(go env GOROOT)/misc/wasm/wasm_exec.js" /build/wasm_exec.js; \
    fi

# Stage 1: Build frontend with Node.js
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY dashboard/frontend/package.json dashboard/frontend/package-lock.json dashboard/frontend/tsconfig.json dashboard/frontend/tsconfig.node.json dashboard/frontend/vite.config.ts ./
COPY dashboard/frontend/src ./src
COPY dashboard/frontend/public ./public
COPY dashboard/frontend/index.html ./
COPY --from=wasm-builder /build/signal-compiler.wasm /build/wasm_exec.js ./public/
# Use official npm registry to avoid mirror timeouts
RUN npm config set registry https://registry.npmjs.org/ && \
    npm install --include=dev  # Ensure TypeScript/Vite are installed regardless of NODE_ENV
RUN npm run build

FROM node:20-alpine AS wizmap-builder
WORKDIR /app/wizmap
COPY dashboard/wizmap/package.json dashboard/wizmap/package-lock.json ./
RUN npm config set registry https://registry.npmjs.org/ && \
    npm ci --include=dev
COPY dashboard/wizmap/ ./
RUN npm run build -- --outDir /app/wizmap-dist

# Stage 2: Build backend with Go
FROM golang:1.24-alpine AS backend-builder
WORKDIR /app

# Use Go proxy for reliable downloads
ENV GOPROXY=https://proxy.golang.org,direct

# Accept build arguments for target architecture
ARG TARGETARCH
ARG TARGETOS=linux

RUN set -eux; \
    apk_add_with_retry() { \
      for attempt in 1 2 3 4 5; do \
        if apk add --no-cache "$@"; then \
          return 0; \
        fi; \
        echo "apk add failed (attempt ${attempt}/5), retrying..." >&2; \
        sleep $((attempt * 2)); \
      done; \
      return 1; \
    }; \
    apk_add_with_retry build-base

# Copy src/semantic-router first (required by dashboard/backend/go.mod replace directive)
# Only copy the packages we actually need for dashboard backend builds - NOT pkg/cli which has extra deps
COPY src/semantic-router/go.mod src/semantic-router/go.sum /app/src/semantic-router/
COPY src/semantic-router/pkg/config/ /app/src/semantic-router/pkg/config/
COPY src/semantic-router/pkg/dsl/ /app/src/semantic-router/pkg/dsl/
COPY src/semantic-router/pkg/modelinventory/ /app/src/semantic-router/pkg/modelinventory/
COPY src/semantic-router/pkg/observability/ /app/src/semantic-router/pkg/observability/
COPY src/semantic-router/pkg/startupstatus/ /app/src/semantic-router/pkg/startupstatus/

# Copy go.mod and go.sum first for better caching
COPY dashboard/backend/go.mod dashboard/backend/go.sum /app/dashboard/backend/
WORKDIR /app/dashboard/backend
RUN go mod download

# Copy source code and build
COPY dashboard/backend/ /app/dashboard/backend/
RUN CGO_ENABLED=1 GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build -ldflags="-w -s" -o /app/dashboard-backend main.go

# Stage 3: Final runtime image
FROM alpine:3.19
ENV PYTHONPATH=/app
ENV VIRTUAL_ENV=/opt/vllm-sr-dashboard-venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
RUN set -eux; \
    apk_add_with_retry() { \
      for attempt in 1 2 3 4 5; do \
        if apk add --no-cache "$@"; then \
          return 0; \
        fi; \
        echo "apk add failed (attempt ${attempt}/5), retrying..." >&2; \
        sleep $((attempt * 2)); \
      done; \
      return 1; \
    }; \
    apk_add_with_retry ca-certificates curl docker-cli py3-pip python3 su-exec wget
WORKDIR /app
COPY --from=backend-builder /app/dashboard-backend /app/dashboard-backend
COPY --from=backend-builder /app/dashboard-backend /usr/local/bin/dashboard-backend
COPY --from=frontend-builder /app/frontend/dist /app/frontend
COPY --from=wizmap-builder /app/wizmap-dist /app/frontend/embedded/wizmap
COPY src/vllm-sr/requirements.txt /app/requirements.txt
COPY src/vllm-sr/cli/ /app/cli/
COPY src/vllm-sr/start-dashboard.sh /app/start-dashboard.sh
COPY src/vllm-sr/cli/templates/tools_db.json /app/config/tools_db.json
COPY config/kb/ /app/config/kb/
COPY dashboard/backend/config/openclaw-skills.json /app/config/openclaw-skills.json
COPY dashboard/backend/skillpacks/ /app/skills/
# Copy entrypoint script
COPY dashboard/backend/entrypoint.sh /app/entrypoint.sh
RUN python3 -m venv "${VIRTUAL_ENV}" && \
    "${VIRTUAL_ENV}/bin/pip" install --no-cache-dir --upgrade pip && \
    "${VIRTUAL_ENV}/bin/pip" install --no-cache-dir -r /app/requirements.txt && \
    chmod +x /app/entrypoint.sh /app/start-dashboard.sh
RUN addgroup -g 65532 nonroot && \
    adduser -D -u 65532 -G nonroot nonroot && \
    chown -R nonroot:nonroot /app
EXPOSE 8700
# Note: We run as root initially so entrypoint can fix ownership/permissions, then switch to nonroot
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["/app/dashboard-backend", "-port=8700", "-static=/app/frontend", "-config=/app/config/config.yaml"]
