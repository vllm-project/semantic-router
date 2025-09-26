# ====================== common.mk ======================
# = Common function or variables for other makefiles    =
# ====================== common.mk ======================

# Turn off .INTERMEDIATE file removal by marking all files as
# .SECONDARY.  .INTERMEDIATE file removal is a space-saving hack from
# a time when drives were small; on modern computers with plenty of
# storage, it causes nothing but headaches.
#
# https://news.ycombinator.com/item?id=16486331
.SECONDARY:

# Variables Define
DATETIME = $(shell date +"%Y%m%d%H%M%S")

# REV is the short git sha of latest commit.
REV=$(shell git rev-parse --short HEAD)

# Function Define

# logging Output Function
# Log normal info
LOG_TARGET = echo "\033[0;32m==================> Running $@ ============> ... \033[0m"

# Log debugging info
define log
echo "\033[36m==================>$1\033[0m"
endef

# Log error info
define errorLog
echo "\033[0;31m==================>$1\033[0m"
endef

# Help target
help:
	@echo "\033[1;3;34mIntelligent Mixture-of-Models Router for Efficient LLM Inference.\033[0m\n"
	@echo "Available targets:"
	@echo "  Build targets:"
	@echo "    all                     - Build everything (default)"
	@echo "    build                   - Build Rust library and Go router"
	@echo "    rust                    - Build only the Rust library"
	@echo "    build-router            - Build only the Go router"
	@echo "    clean                   - Clean build artifacts"
	@echo ""
	@echo "  Run targets:"
	@echo "    run-router              - Run the router (CONFIG_FILE=config/config.yaml)"
	@echo "    run-router-e2e          - Run the router with e2e config (config/config.e2e.yaml)"
	@echo "    run-envoy               - Run Envoy proxy"
	@echo ""
	@echo "  Test targets:"
	@echo "    test                    - Run all tests"
	@echo "    test-binding            - Test candle-binding"
	@echo "    test-semantic-router    - Test semantic router"
	@echo "    test-category-classifier - Test category classifier"
	@echo "    test-pii-classifier     - Test PII classifier"
	@echo "    test-jailbreak-classifier - Test jailbreak classifier"
	@echo ""
	@echo "  E2E Test targets:"
	@echo "    start-llm-katan         - Start LLM Katan servers for e2e tests"
	@echo "    test-e2e-vllm           - Run e2e tests with LLM Katan servers"
	@echo ""
	@echo "  Milvus targets (CONTAINER_RUNTIME=docker|podman):"
	@echo "    start-milvus            - Start Milvus container for testing"
	@echo "    stop-milvus             - Stop and remove Milvus container"
	@echo "    restart-milvus          - Restart Milvus container"
	@echo "    milvus-status           - Check Milvus container status"
	@echo "    clean-milvus            - Stop container and clean data"
	@echo "    test-milvus-cache       - Test cache with Milvus backend"
	@echo "    test-semantic-router-milvus - Test router with Milvus cache"
	@echo "    start-milvus-ui         - Start Milvus UI to browse data"
	@echo "    stop-milvus-ui          - Stop and remove Milvus UI container"
	@echo "    Example: CONTAINER_RUNTIME=podman make start-milvus"
	@echo ""
	@echo "  Demo targets:"
	@echo "    test-auto-prompt-reasoning - Test reasoning mode"
	@echo "    test-auto-prompt-no-reasoning - Test normal mode"
	@echo "    test-pii                - Test PII detection"
	@echo "    test-prompt-guard       - Test jailbreak detection"
	@echo "    test-tools              - Test tool auto-selection"
	@echo ""
	@echo "  Documentation targets:"
	@echo "    docs-dev                - Start documentation dev server"
	@echo "    docs-build              - Build documentation"
	@echo "    docs-serve              - Serve built documentation"
	@echo "    docs-clean              - Clean documentation artifacts"
	@echo ""
	@echo "  Environment variables:"
	@echo "    CONTAINER_RUNTIME       - Container runtime (docker|podman, default: docker)"
	@echo "    CONFIG_FILE             - Config file path (default: config/config.yaml)"
	@echo "    VLLM_ENDPOINT           - vLLM endpoint URL for testing"
	@echo ""
	@echo "  Usage examples:"
	@echo "    make start-milvus                    # Use Docker (default)"
	@echo "    CONTAINER_RUNTIME=podman make start-milvus  # Use Podman"
	@echo "    CONFIG_FILE=custom.yaml make run-router     # Use custom config"
