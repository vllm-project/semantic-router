FROM golang:1.24 AS builder
WORKDIR /app
# Copy only module files first for layer caching
COPY dashboard/backend/go.mod dashboard/backend/go.sum ./dashboard/backend/
WORKDIR /app/dashboard/backend
RUN go mod download
# Copy source and frontend assets
COPY dashboard/backend/ ./
COPY dashboard/frontend/ /app/dashboard/frontend/
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o /app/dashboard-backend main.go

FROM gcr.io/distroless/static:nonroot
WORKDIR /app
COPY --from=builder /app/dashboard-backend /app/dashboard-backend
COPY --from=builder /app/dashboard/frontend /app/frontend
EXPOSE 8700
USER nonroot:nonroot
ENTRYPOINT ["/app/dashboard-backend", "-port=8700", "-static=/app/frontend"]
