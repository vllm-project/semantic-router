# ================= performance.mk =================
# Manifest-driven performance runs and reports.
# ==================================================

##@ Performance Testing

PERF_ENV ?= cpu
PERF_PROFILE ?= quick
PERF_CONFIG ?= $(CURDIR)/perf/config/perf.yaml
PERF_THRESHOLDS ?= $(CURDIR)/perf/config/thresholds.yaml
PERF_OUTPUT_DIR ?= $(CURDIR)/reports/perf/$(PERF_ENV)-$(PERF_PROFILE)
PERF_BASELINE ?= $(CURDIR)/perf/testdata/baselines/$(PERF_ENV)-ci.json
PERF_GATE_FLAGS ?=

.PHONY: ensure-reports-dir
ensure-reports-dir:
	@mkdir -p "$(PERF_OUTPUT_DIR)"

perf-validate: ## Validate the manifest and threshold policy
	@$(LOG_TARGET)
	@cd perf && go run ./cmd/perftest validate \
		--config "$(PERF_CONFIG)" \
		--thresholds "$(PERF_THRESHOLDS)"

perf-unit: ## Run performance harness unit tests
	@$(LOG_TARGET)
	@cd perf && go test ./pkg/benchmark/... ./cmd/perftest/...

# The actual benchmark functions live beside the package hot paths. build-router
# prepares the cgo/native libraries those packages link against; the manifest
# chooses the environment, profile, suite inventory, repetition count, and
# benchmark duration. Every run writes current/comparison/report JSON plus
# Markdown, HTML, and per-suite raw logs under PERF_OUTPUT_DIR.
perf-run: rust-ci perf-validate ensure-reports-dir ## Run a manifest profile and generate reports
	@$(LOG_TARGET)
	@export LD_LIBRARY_PATH="$(CURDIR)/candle-binding/target/release:$(CURDIR)/ml-binding/target/release:$(CURDIR)/nlp-binding/target/release:$${LD_LIBRARY_PATH}"; \
	cd perf && go run ./cmd/perftest run \
		--config "$(PERF_CONFIG)" \
		--thresholds "$(PERF_THRESHOLDS)" \
		--repo-root "$(CURDIR)" \
		--environment "$(PERF_ENV)" \
		--profile "$(PERF_PROFILE)" \
		--output-dir "$(PERF_OUTPUT_DIR)" \
		$(if $(strip $(PERF_BASELINE)),--baseline "$(PERF_BASELINE)",) \
		$(PERF_GATE_FLAGS)

perf-bench-quick: PERF_PROFILE = quick
perf-bench-quick: perf-run ## Fast local CPU hot-path report

perf-check: PERF_PROFILE = ci
perf-check: PERF_GATE_FLAGS = --fail-on-regression --require-complete
perf-check: perf-run ## Run the fail-closed CPU PR performance gate

perf-nightly: PERF_PROFILE = nightly
perf-nightly: PERF_GATE_FLAGS = --fail-on-regression --require-complete
perf-nightly: perf-run ## Run the longer CPU trend profile

perf-bench: PERF_PROFILE = cpu-full
perf-bench: PERF_GATE_FLAGS =
perf-bench: perf-run ## Run all currently available CPU suites (models required)

perf-compare: ensure-reports-dir ## Recompare an existing current.json and regenerate reports
	@$(LOG_TARGET)
	@cd perf && go run ./cmd/perftest compare \
		--current "$(PERF_OUTPUT_DIR)/current.json" \
		--baseline "$(PERF_BASELINE)" \
		--thresholds "$(PERF_THRESHOLDS)" \
		--output-dir "$(PERF_OUTPUT_DIR)"

perf-report: ensure-reports-dir ## Regenerate JSON, Markdown, and HTML from comparison.json
	@$(LOG_TARGET)
	@cd perf && go run ./cmd/perftest report \
		--input "$(PERF_OUTPUT_DIR)/comparison.json" \
		--output-dir "$(PERF_OUTPUT_DIR)"

# Baselines are never updated by scheduled CI. This explicit local command
# captures the CPU CI inventory, then promotes the reviewed current result.
perf-baseline-update: PERF_PROFILE = ci
perf-baseline-update: PERF_BASELINE =
perf-baseline-update: perf-run ## Capture and promote a reviewed CPU CI baseline
	@$(LOG_TARGET)
	@cd perf && go run ./cmd/perftest promote \
		--current "$(PERF_OUTPUT_DIR)/current.json" \
		--output "$(CURDIR)/perf/testdata/baselines/$(PERF_ENV)-ci.json"

perf-clean: ## Remove generated performance artifacts
	@$(LOG_TARGET)
	@rm -rf "$(CURDIR)/reports/perf"

perf-help: ## Show performance framework commands and overrides
	@echo "Performance framework:"
	@echo "  make perf-bench-quick                 fast local CPU report"
	@echo "  make perf-check                       fail-closed CPU PR gate"
	@echo "  make perf-nightly                     longer CPU trend run"
	@echo "  make perf-bench                       opt-in full CPU/model run"
	@echo "  make perf-baseline-update             explicitly promote CPU CI baseline"
	@echo "  make perf-compare PERF_OUTPUT_DIR=... recompare captured results"
	@echo ""
	@echo "Overrides: PERF_ENV, PERF_PROFILE, PERF_OUTPUT_DIR, PERF_BASELINE"
