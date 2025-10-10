FROM golang:1.24 AS builder
WORKDIR /app
# Copy source and frontend assets
COPY dashboard/backend/ /app/dashboard/backend/
COPY dashboard/frontend/ /app/dashboard/frontend/
# Build the binary
WORKDIR /app/dashboard/backend
RUN go mod download || true
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o /app/dashboard-backend main.go

FROM alpine:3.19
RUN apk --no-cache add ca-certificates wget
WORKDIR /app
COPY --from=builder /app/dashboard-backend /app/dashboard-backend
COPY --from=builder /app/dashboard/frontend /app/frontend
RUN addgroup -g 65532 nonroot && \
    adduser -D -u 65532 -G nonroot nonroot && \
    chown -R nonroot:nonroot /app
EXPOSE 8700
USER nonroot:nonroot
ENTRYPOINT ["/app/dashboard-backend", "-port=8700", "-static=/app/frontend"]
