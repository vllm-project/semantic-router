# cmd/wasm/Makefile — Build DSL compiler to WebAssembly
#
# Usage:
#   make build          Build WASM binary + copy wasm_exec.js
#   make build-tiny     Build with TinyGo for smaller binary (optional)
#   make clean          Remove build artifacts
#   make size           Show WASM binary size

WASM_OUT      = ../../dashboard/frontend/public/signal-compiler.wasm
WASM_EXEC_JS  = ../../dashboard/frontend/public/wasm_exec.js
MODULE_ROOT   = ../..

.PHONY: build build-tiny clean size

build:
	@echo "Building WASM binary..."
	cd $(MODULE_ROOT) && CGO_ENABLED=0 GOOS=js GOARCH=wasm go build \
		-o $(CURDIR)/$(WASM_OUT) \
		-ldflags="-s -w" \
		./cmd/wasm/
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
	cd $(MODULE_ROOT) && tinygo build \
		-o $(CURDIR)/$(WASM_OUT) \
		-target wasm \
		-no-debug \
		./cmd/wasm/
	@cp "$$(tinygo env TINYGOROOT)/targets/wasm_exec.js" $(WASM_EXEC_JS)
	@echo "WASM size: $$(du -h $(WASM_OUT) | cut -f1)"

clean:
	rm -f $(WASM_OUT) $(WASM_EXEC_JS)

size:
	@if [ -f $(WASM_OUT) ]; then \
		echo "WASM binary: $$(du -h $(WASM_OUT) | cut -f1)"; \
		echo "wasm_exec.js: $$(du -h $(WASM_EXEC_JS) | cut -f1)"; \
	else \
		echo "WASM binary not found. Run 'make build' first."; \
	fi
