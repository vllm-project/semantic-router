# Build frontend
FROM node:18-alpine AS frontend-builder
ARG BASE_DIR=dashboard
WORKDIR /app/frontend
COPY ${BASE_DIR}/frontend/package*.json ./
# Use official npm registry to avoid mirror timeouts
RUN npm config set registry https://registry.npmjs.org/ && \
    npm ci --include=dev   # Ensure devDependencies are installed regardless of NODE_ENV
COPY ${BASE_DIR}/frontend/ ./
RUN npm run build

# Build backend (needs src/semantic-router for replace directive)
FROM golang:1.24.1-alpine AS backend-builder
ARG BASE_DIR=dashboard
ARG ROUTER_DIR=src/semantic-router
WORKDIR /app
COPY ${ROUTER_DIR} /src/semantic-router
WORKDIR /app/backend
COPY ${BASE_DIR}/backend/go.* ./
RUN go mod download
COPY ${BASE_DIR}/backend/ ./
RUN CGO_ENABLED=0 GOOS=linux go build -o dashboard-server .

# Final image
FROM alpine:3.18
RUN apk add --no-cache ca-certificates
WORKDIR /app
COPY --from=backend-builder /app/backend/dashboard-server .
COPY --from=frontend-builder /app/frontend/dist ./frontend
ENV DASHBOARD_STATIC_DIR=./frontend
EXPOSE 8700
CMD ["./dashboard-server"]
