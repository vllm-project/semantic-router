# Focused CPU checks for the ONNX replacement of the Candle Go API.

.PHONY: test-onnx-binding test-onnx-router-build

test-onnx-binding: build-onnx-binding ## Test ONNX tokenizer windows and the real Go/FFI boundary without models
	@cd onnx-binding && \
		cargo test --release --locked --no-default-features --lib text_windows
	@cd onnx-binding && \
		LD_LIBRARY_PATH="$(CURDIR)/onnx-binding/target/release:$${LD_LIBRARY_PATH:-}" \
		CGO_ENABLED=1 go test -count=1 -v semantic-router.go text_windows_test.go

test-onnx-router-build: build-onnx-binding build-ml-binding ## Compile the router against the ONNX replacement module
	@cd nlp-binding && cargo build --release --locked
	@mkdir -p bin
	@cd src/semantic-router && \
		CGO_ENABLED=1 \
		CGO_LDFLAGS="-L$(CURDIR)/onnx-binding/target/release -L$(CURDIR)/ml-binding/target/release -L$(CURDIR)/nlp-binding/target/release" \
		go build -modfile=go.onnx.mod -tags=onnx -o ../../bin/router-onnx ./cmd
