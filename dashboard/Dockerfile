# Build frontend
FROM node:18-alpine AS frontend-builder
ENV http_proxy=http://proxy.iil.intel.com:911
ENV https_proxy=http://proxy.iil.intel.com:911
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Build backend
FROM golang:1.21-alpine AS backend-builder
ENV http_proxy=http://proxy.iil.intel.com:911
ENV https_proxy=http://proxy.iil.intel.com:911
WORKDIR /app/backend
COPY backend/go.* ./
RUN go mod download
COPY backend/ ./
RUN CGO_ENABLED=0 GOOS=linux go build -o dashboard-server .

# Final image
FROM alpine:3.18
ENV http_proxy=http://proxy.iil.intel.com:911
ENV https_proxy=http://proxy.iil.intel.com:911
RUN apk add --no-cache ca-certificates
WORKDIR /app
COPY --from=backend-builder /app/backend/dashboard-server .
COPY --from=frontend-builder /app/frontend/dist ./frontend
ENV DASHBOARD_STATIC_DIR=./frontend
EXPOSE 8700
CMD ["./dashboard-server"]
