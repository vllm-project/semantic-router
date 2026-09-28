# Build the Dashboard DSL compiler with the Router module's shared dependencies.
#
# Usage:
#   make build          Build WASM binary + copy wasm_exec.js
#   make build-tiny     Build with TinyGo for smaller binary (optional)
#   make test           Run Go WASM tests and Node integration assertions
#   make clean          Remove build artifacts
#   make size           Show WASM binary size

WASM_OUT      = $(CURDIR)/../frontend/public/signal-compiler.wasm
WASM_EXEC_JS  = $(CURDIR)/../frontend/public/wasm_exec.js
MODULE_ROOT   = $(CURDIR)/../../src/semantic-router
WASM_SOURCE   = $(CURDIR)/main_wasm.go
WASM_TEST     = $(CURDIR)/wasm_test.go
WASM_REPORT_DIR ?= $(CURDIR)/../../.agent-harness/dashboard/wasm
PYTHON ?= python3

.PHONY: build build-tiny test test-built clean size

build:
	@echo "Building WASM binary..."
	@mkdir -p $(dir $(WASM_OUT))
	cd $(MODULE_ROOT) && CGO_ENABLED=0 GOOS=js GOARCH=wasm go build \
		-buildvcs=false \
		-o $(WASM_OUT) \
		-ldflags="-s -w" \
		$(WASM_SOURCE)
	@# Go 1.24+ moved wasm_exec.js from misc/wasm/ to lib/wasm/
	@if [ -f "$$(go env GOROOT)/lib/wasm/wasm_exec.js" ]; then \
		cp "$$(go env GOROOT)/lib/wasm/wasm_exec.js" $(WASM_EXEC_JS); \
	else \
		cp "$$(go env GOROOT)/misc/wasm/wasm_exec.js" $(WASM_EXEC_JS); \
	fi
	@echo "WASM size: $$(du -h $(WASM_OUT) | cut -f1)"
	@echo "Output: $(WASM_OUT)"
	@echo "Glue:   $(WASM_EXEC_JS)"

build-tiny:
	@echo "Building WASM binary with TinyGo (smaller size)..."
	@mkdir -p $(dir $(WASM_OUT))
	cd $(MODULE_ROOT) && tinygo build \
		-o $(WASM_OUT) \
		-target wasm \
		-no-debug \
		$(WASM_SOURCE)
	@cp "$$(tinygo env TINYGOROOT)/targets/wasm_exec.js" $(WASM_EXEC_JS)
	@echo "WASM size: $$(du -h $(WASM_OUT) | cut -f1)"

test: build
	@$(MAKE) test-built

# The root Dashboard targets share one build prerequisite across tests/builds.
test-built:
	@mkdir -p "$(WASM_REPORT_DIR)"
	@rm -f "$(WASM_REPORT_DIR)/evidence.json" "$(WASM_REPORT_DIR)/node-tests.jsonl"
	cd $(MODULE_ROOT) && CGO_ENABLED=0 GOOS=js GOARCH=wasm go test -json \
		-exec="$(CURDIR)/go_test_exec.sh" \
		-list '^(Test|Fuzz|Example)' $(WASM_SOURCE) $(WASM_TEST) \
		> "$(WASM_REPORT_DIR)/inventory.jsonl"
	@if cd $(MODULE_ROOT) && CGO_ENABLED=0 GOOS=js GOARCH=wasm go test -json -count=1 \
		-exec="$(CURDIR)/go_test_exec.sh" \
		$(WASM_SOURCE) $(WASM_TEST) > "$(WASM_REPORT_DIR)/go-tests.jsonl"; then \
		cat "$(WASM_REPORT_DIR)/go-tests.jsonl"; \
	else cat "$(WASM_REPORT_DIR)/go-tests.jsonl"; exit 1; fi
	DASHBOARD_WASM_NODE_EVENTS="$(WASM_REPORT_DIR)/node-tests.jsonl" \
		node $(CURDIR)/wasm_node_test.js $(WASM_OUT) $(WASM_EXEC_JS)
	$(PYTHON) $(CURDIR)/../../tools/ci/dashboard_wasm_evidence.py \
		--report "$(WASM_REPORT_DIR)" --node-source $(CURDIR)/wasm_node_test.js

clean:
	rm -f $(WASM_OUT) $(WASM_EXEC_JS)

size:
	@if [ -f $(WASM_OUT) ]; then \
		echo "WASM binary: $$(du -h $(WASM_OUT) | cut -f1)"; \
		echo "wasm_exec.js: $$(du -h $(WASM_EXEC_JS) | cut -f1)"; \
	else \
		echo "WASM binary not found. Run 'make build' first."; \
	fi
