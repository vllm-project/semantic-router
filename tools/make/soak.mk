##@ Soak Testing

SOAK_CONFIG ?=
SOAK_DELAY_MS ?= 1500
SOAK_DELAY_JITTER_MS ?= 500
SOAK_STREAMING ?= 0
SOAK_STREAM_TTFT_MS ?= 500
SOAK_STREAM_TTFT_JITTER_MS ?= 0
SOAK_STREAM_INTERVAL_MS ?= 50
SOAK_STREAM_FRAMES ?= 64
SOAK_LOG_DIR ?= /tmp/soak-logs
SOAK_ARGS ?=

SOAK_SCRIPT := tools/soak/run-soak-local.sh
SOAK_ENV := SOAK_CONFIG=$(SOAK_CONFIG) \
	SOAK_DELAY_MS=$(SOAK_DELAY_MS) \
	SOAK_DELAY_JITTER_MS=$(SOAK_DELAY_JITTER_MS) \
	SOAK_STREAMING=$(SOAK_STREAMING) \
	SOAK_STREAM_TTFT_MS=$(SOAK_STREAM_TTFT_MS) \
	SOAK_STREAM_TTFT_JITTER_MS=$(SOAK_STREAM_TTFT_JITTER_MS) \
	SOAK_STREAM_INTERVAL_MS=$(SOAK_STREAM_INTERVAL_MS) \
	SOAK_STREAM_FRAMES=$(SOAK_STREAM_FRAMES) \
	SOAK_LOG_DIR=$(SOAK_LOG_DIR)

build-soak: ## Build the soak harness binary
build-soak:
	@$(LOG_TARGET)
	@mkdir -p bin
	@cd e2e && go build -o ../bin/soak ./cmd/soak

soak-test: ## Vet and unit-test the soak harness (no running stack required)
soak-test:
	@$(LOG_TARGET)
	@cd e2e && go build ./... && go vet ./cmd/soak/... ./pkg/soak/... && go test -count=1 ./pkg/soak/...

soak-local: ## Run the local soak baseline against a full router + Envoy + mock backend stack
soak-local: build-router build-soak
	@$(LOG_TARGET)
	@$(SOAK_ENV) $(SOAK_SCRIPT) $(SOAK_ARGS)

soak-help: ## Print the soak harness flags and the knobs this Makefile exposes
soak-help: build-soak
	@$(LOG_TARGET)
	@echo "Make variables:"
	@echo "  SOAK_CONFIG=$(SOAK_CONFIG) (empty: derived from e2e/config/config.e2e.yaml)"
	@echo "  SOAK_DELAY_MS=$(SOAK_DELAY_MS)"
	@echo "  SOAK_DELAY_JITTER_MS=$(SOAK_DELAY_JITTER_MS)"
	@echo "  SOAK_STREAMING=$(SOAK_STREAMING) (1: request and validate SSE responses)"
	@echo "  SOAK_STREAM_TTFT_MS=$(SOAK_STREAM_TTFT_MS)"
	@echo "  SOAK_STREAM_TTFT_JITTER_MS=$(SOAK_STREAM_TTFT_JITTER_MS)"
	@echo "  SOAK_STREAM_INTERVAL_MS=$(SOAK_STREAM_INTERVAL_MS)"
	@echo "  SOAK_STREAM_FRAMES=$(SOAK_STREAM_FRAMES) (content frames before terminal events)"
	@echo "  SOAK_LOG_DIR=$(SOAK_LOG_DIR)"
	@echo "  SOAK_ARGS=$(SOAK_ARGS)"
	@echo ""
	@echo "Harness flags (forwarded via SOAK_ARGS):"
	@./bin/soak -h 2>&1 || true
