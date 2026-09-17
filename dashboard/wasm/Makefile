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

.PHONY: build build-tiny test clean size

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
	cd $(MODULE_ROOT) && CGO_ENABLED=0 GOOS=js GOARCH=wasm go test \
		-exec="$$(go env GOROOT)/lib/wasm/go_js_wasm_exec" \
		$(WASM_SOURCE) $(WASM_TEST)
	node $(CURDIR)/wasm_node_test.js $(WASM_OUT) $(WASM_EXEC_JS)

clean:
	rm -f $(WASM_OUT) $(WASM_EXEC_JS)

size:
	@if [ -f $(WASM_OUT) ]; then \
		echo "WASM binary: $$(du -h $(WASM_OUT) | cut -f1)"; \
		echo "wasm_exec.js: $$(du -h $(WASM_EXEC_JS) | cut -f1)"; \
	else \
		echo "WASM binary not found. Run 'make build' first."; \
	fi
