# Stage 1: Build frontend with Node.js
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json frontend/tsconfig.json frontend/tsconfig.node.json frontend/vite.config.ts ./
COPY frontend/src ./src
COPY frontend/public ./public
COPY frontend/index.html ./
RUN npm install --include=dev
RUN npm run build

# Stage 2: Build backend with Go
FROM golang:1.24 AS backend-builder
WORKDIR /app

# Use Chinese Go proxy to avoid network timeout issues
ENV GOPROXY=https://goproxy.cn,direct
ENV GOSUMDB=sum.golang.google.cn

# Accept build arguments for target architecture
ARG TARGETARCH
ARG TARGETOS=linux

# Copy go.mod and go.sum first for better caching
COPY backend/go.mod backend/go.sum ./backend/
WORKDIR /app/backend
RUN go mod download

# Copy source code and build
COPY backend/ ./
RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build -ldflags="-w -s" -o /app/dashboard-backend main.go

# Stage 3: Final runtime image
FROM alpine:3.19
RUN apk --no-cache add ca-certificates wget
WORKDIR /app
COPY --from=backend-builder /app/dashboard-backend /app/dashboard-backend
COPY --from=frontend-builder /app/frontend/dist /app/frontend
RUN addgroup -g 65532 nonroot && \
    adduser -D -u 65532 -G nonroot nonroot && \
    chown -R nonroot:nonroot /app
EXPOSE 8700
USER nonroot:nonroot
ENTRYPOINT ["/app/dashboard-backend", "-port=8700", "-static=/app/frontend"]
