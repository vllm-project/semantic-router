# Stage 1: Build frontend with Node.js
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY dashboard/frontend/package.json dashboard/frontend/package-lock.json dashboard/frontend/tsconfig.json dashboard/frontend/tsconfig.node.json dashboard/frontend/vite.config.ts ./
COPY dashboard/frontend/src ./src
COPY dashboard/frontend/public ./public
COPY dashboard/frontend/index.html ./
# Use official npm registry to avoid mirror timeouts
RUN npm config set registry https://registry.npmjs.org/ && \
    npm install --include=dev  # Ensure TypeScript/Vite are installed regardless of NODE_ENV
RUN npm run build

# Stage 2: Build backend with Go
FROM golang:1.24 AS backend-builder
WORKDIR /app

# Use Go proxy for reliable downloads
ENV GOPROXY=https://proxy.golang.org,direct

# Accept build arguments for target architecture
ARG TARGETARCH
ARG TARGETOS=linux

# Copy src/semantic-router first (required by dashboard/backend/go.mod replace directive)
# Only copy the packages we actually need (config, observability) - NOT pkg/cli which has extra deps
COPY src/semantic-router/go.mod src/semantic-router/go.sum /app/src/semantic-router/
COPY src/semantic-router/pkg/config/ /app/src/semantic-router/pkg/config/
COPY src/semantic-router/pkg/observability/ /app/src/semantic-router/pkg/observability/

# Copy go.mod and go.sum first for better caching
COPY dashboard/backend/go.mod dashboard/backend/go.sum /app/dashboard/backend/
WORKDIR /app/dashboard/backend
RUN go mod download

# Copy source code and build
COPY dashboard/backend/ /app/dashboard/backend/
RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build -ldflags="-w -s" -o /app/dashboard-backend main.go

# Stage 3: Final runtime image
FROM alpine:3.19
RUN apk --no-cache add ca-certificates wget su-exec
WORKDIR /app
COPY --from=backend-builder /app/dashboard-backend /app/dashboard-backend
COPY --from=frontend-builder /app/frontend/dist /app/frontend
# Copy entrypoint script
COPY dashboard/backend/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
RUN addgroup -g 65532 nonroot && \
    adduser -D -u 65532 -G nonroot nonroot && \
    chown -R nonroot:nonroot /app
EXPOSE 8700
# Note: We run as root initially so entrypoint can fix ownership/permissions, then switch to nonroot
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["/app/dashboard-backend", "-port=8700", "-static=/app/frontend", "-config=/app/config/config.yaml"]
