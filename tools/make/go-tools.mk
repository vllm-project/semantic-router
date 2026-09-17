# Repository Go tools reuse the Router module's dependency versions. Keeping
# their sources outside the runtime module also means builds, tests and lint
# must select them explicitly; `go test ./...` cannot discover these commands.
# These native commands have no build-tag variants. The Dashboard WASM adapter
# selects its js/wasm sources separately in dashboard/wasm/Makefile.

ROUTER_GO_TOOLS := sr-dsl classifier-operating-point fusioneval image-routing-calibration

sr-dsl_DIR := tools/dev/dsl
classifier-operating-point_DIR := tools/models/classifier-operating-point
fusioneval_DIR := bench/grounded_fusion/fusioneval
image-routing-calibration_DIR := tools/calibration/image-routing

GO_TOOL_BUILD_FLAGS ?=
GO_TOOL_TEST_FLAGS ?=
GO_TOOL_ARGS ?=

go_tool_sources = $(addprefix ../../,$(filter-out %_test.go,$(wildcard $($(1)_DIR)/*.go)))
go_tool_test_sources = $(addprefix ../../,$(wildcard $($(1)_DIR)/*.go))

define router_go_tool
.PHONY: build-$(1) run-$(1) test-$(1) vet-$(1) lint-$(1)

build-$(1):
	@mkdir -p bin
	@cd src/semantic-router && $$(NATIVE_ENV) go build $(GO_TOOL_BUILD_FLAGS) -o ../../bin/$(1) $(call go_tool_sources,$(1))

run-$(1):
	@cd src/semantic-router && $$(NATIVE_ENV) go run $(GO_TOOL_BUILD_FLAGS) $(call go_tool_sources,$(1)) $(GO_TOOL_ARGS)

test-$(1):
	@cd src/semantic-router && $$(NATIVE_ENV) go test $(GO_TOOL_TEST_FLAGS) $(call go_tool_test_sources,$(1))

vet-$(1):
	@cd src/semantic-router && $$(NATIVE_ENV) go vet $(call go_tool_test_sources,$(1))

lint-$(1):
	@cd src/semantic-router && $$(NATIVE_ENV) golangci-lint run --config ../../tools/linter/go/.golangci.yml $(call go_tool_test_sources,$(1))
endef

$(foreach tool,$(ROUTER_GO_TOOLS),$(eval $(call router_go_tool,$(tool))))

go-tools-build: $(addprefix build-,$(ROUTER_GO_TOOLS)) ## Build repository Go tools with Router module dependencies
go-tools-test: $(addprefix test-,$(ROUTER_GO_TOOLS)) ## Test repository Go tools, including offline calibration logic
go-tools-vet: $(addprefix vet-,$(ROUTER_GO_TOOLS)) ## Vet repository Go tools
go-tools-lint: $(addprefix lint-,$(ROUTER_GO_TOOLS)) ## Lint repository Go tools

.PHONY: go-tools-build go-tools-test go-tools-vet go-tools-lint
