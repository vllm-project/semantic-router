# Build frontend
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Build backend
FROM golang:1.21-alpine AS backend-builder
WORKDIR /app/backend
COPY backend/go.* ./
RUN go mod download
COPY backend/ ./
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
