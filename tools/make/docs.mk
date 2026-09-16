# ========================== docs.mk ==========================
# = Everything For Docs,include API Docs and Docs Website     =
# ========================== docs.mk ==========================

##@ Docs

DOCS_TRANSLATION_LOCALE ?= zh-Hans
DOCS_VENV ?= $(CURDIR)/website/.venv
DOCS_VENV_PYTHON = $(DOCS_VENV)/bin/python
DOCS_PYTHON ?= $(if $(wildcard $(DOCS_VENV_PYTHON)),$(DOCS_VENV_PYTHON),$(if $(wildcard $(CURDIR)/.venv-agent/bin/python),$(CURDIR)/.venv-agent/bin/python,python3))

.PHONY: docs-cli docs-cli-check docs-cli-test docs-community-check docs-community-test docs-generated-check
docs-cli: ## Generate the CLI command reference from the registered Click command tree
	@$(DOCS_PYTHON) tools/docs/generate_cli_reference.py

docs-cli-check: ## Reject a stale CLI command reference without rewriting it
	@$(DOCS_PYTHON) tools/docs/generate_cli_reference.py --check

docs-cli-test: ## Test command discovery and CLI reference drift detection
	@$(DOCS_PYTHON) -m unittest discover -s tools/docs/tests -p 'test_*.py'

docs-community-check: ## Check source provenance of the published GitHub statistics snapshots offline
	@node website/scripts/generate-contributor-rank.mjs --check-source
	@node website/scripts/generate-committer-activity.mjs --check-source

docs-community-test: ## Test offline snapshot provenance and stale-refresh rejection
	@node --test website/scripts/lib/generated-source.test.mjs

docs-generated-check: MODEL_CATALOG_PYTHON = $(DOCS_PYTHON)
docs-generated-check: model-catalog-generated-check docs-cli-check docs-config-check docs-community-check agent-skill-check ## Check generated website contracts without native builds or rewriting

.PHONY: docs-python-install docs-install docs-build
docs-python-install: ## Install generated-reference dependencies in an isolated documentation environment
	@if [ ! -x "$(DOCS_VENV_PYTHON)" ]; then \
		"$(DOCS_PYTHON)" -m venv "$(DOCS_VENV)"; \
	fi
	@"$(DOCS_VENV_PYTHON)" -m pip install --disable-pip-version-check -r tools/docs/requirements.txt

docs-install: docs-python-install ## Install documentation website dependencies
	@$(LOG_TARGET)
	cd website && npm install

docs-dev: docs-install ## Start documentation website in dev mode
	@$(LOG_TARGET)
	cd website && VLLM_SR_DOCS_PYTHON="$(DOCS_VENV_PYTHON)" npm start

docs-dev-zh: docs-install ## Start documentation website in dev mode
	@$(LOG_TARGET)
	cd website && VLLM_SR_DOCS_PYTHON="$(DOCS_VENV_PYTHON)" npm run start:zh

docs-build: docs-install ## Build static documentation website
	@$(LOG_TARGET)
	cd website && VLLM_SR_DOCS_PYTHON="$(DOCS_VENV_PYTHON)" npm run build

docs-serve: docs-build ## Serve built documentation website
	@$(LOG_TARGET)
	cd website && npm run serve

docs-clean: ## Clean documentation build artifacts
	@$(LOG_TARGET)
	cd website && npm run clear

docs-lint: ## Lint documentation website source files
	@$(LOG_TARGET)
	cd website && npm run lint

docs-lint-fix: ## Fix lint issues in documentation website source files
	@$(LOG_TARGET)
	cd website && npm run lint:fix

docs-config: ## Generate the configuration capability catalog
	@$(LOG_TARGET)
	cd website && npm run config:generate

docs-config-check: ## Check that the generated configuration capability catalog is current
	@$(LOG_TARGET)
	cd website && npm run config:check

docs-contributors-rank: ## Generate contributor leaderboard data
	@$(LOG_TARGET)
	cd website && npm run contributors:rank

docs-check-translations: ## Audit documentation translation coverage, metadata, and source drift
	@$(LOG_TARGET)
	website/scripts/check-translation-sync.sh --locale $(DOCS_TRANSLATION_LOCALE)

docs-check-translation-coverage: ## Guard current documentation locale override coverage
	@$(LOG_TARGET)
	website/scripts/check-translation-sync.sh --coverage-only

docs-test-translation-sync: ## Test documentation translation status synchronization
	@$(LOG_TARGET)
	website/scripts/check-translation-sync.test.sh

docs-update-translation-baseline: ## Record current documentation locale override paths for review
	@$(LOG_TARGET)
	website/scripts/check-translation-sync.sh --update-baseline

docs-fix-translation-status: ## Update unambiguous documentation translation outdated flags
	@$(LOG_TARGET)
	@website/scripts/check-translation-sync.sh --locale $(DOCS_TRANSLATION_LOCALE) --fix-status; \
	exit_code=$$?; \
	if [ $$exit_code -ne 0 ] && [ $$exit_code -ne 1 ]; then exit $$exit_code; fi

##@ CRD Documentation

CRD_REF_DOCS_VERSION ?= v0.3.0
CRD_REF_DOCS_BIN ?= $(TOOLS_BIN_DIR)/crd-ref-docs-$(CRD_REF_DOCS_VERSION)

.PHONY: install-crd-ref-docs
install-crd-ref-docs: ## Install crd-ref-docs tool
	@$(LOG_TARGET)
	@if [ ! -x "$(CRD_REF_DOCS_BIN)" ]; then \
		echo "Installing crd-ref-docs..."; \
		tmp_dir=$$(mktemp -d); \
		trap 'rm -rf -- "$$tmp_dir"' EXIT; \
		GOBIN="$$tmp_dir" go install github.com/elastic/crd-ref-docs@$(CRD_REF_DOCS_VERSION); \
		mkdir -p "$(dir $(CRD_REF_DOCS_BIN))"; \
		install -m 0755 "$$tmp_dir/crd-ref-docs" "$(CRD_REF_DOCS_BIN)"; \
	else \
		echo "crd-ref-docs $(CRD_REF_DOCS_VERSION) is already installed at $(CRD_REF_DOCS_BIN)"; \
	fi

.PHONY: docs-crd
docs-crd: install-crd-ref-docs ## Generate CRD API reference documentation
	@$(LOG_TARGET)
	@CRD_REF_DOCS_BIN="$(CRD_REF_DOCS_BIN)" \
		tools/crd/generate-reference.sh website/docs/api/crd-reference.md

.PHONY: docs-crd-check
docs-crd-check: install-crd-ref-docs ## Check that generated CRD documentation is current
	@$(LOG_TARGET)
	@set -eu; \
	tmp_file=$$(mktemp); \
	trap 'rm -f -- "$$tmp_file"' EXIT; \
	CRD_REF_DOCS_BIN="$(CRD_REF_DOCS_BIN)" \
		tools/crd/generate-reference.sh "$$tmp_file"; \
	if ! cmp -s website/docs/api/crd-reference.md "$$tmp_file"; then \
		echo "Generated CRD reference is out of date. Run make docs-crd."; \
		diff -u website/docs/api/crd-reference.md "$$tmp_file" || true; \
		exit 1; \
	fi

.PHONY: docs-crd-watch
docs-crd-watch: ## Watch for CRD changes and regenerate documentation
	@$(LOG_TARGET)
	@echo "Watching for CRD changes..."
	@while true; do \
		$(MAKE) docs-crd; \
		sleep 5; \
	done

.PHONY: docs-all
docs-all: docs-crd docs-config docs-build ## Generate all documentation (CRD + configuration catalog + website)
	@$(LOG_TARGET)
	@echo "All documentation generated successfully"

##@ Apiserver API Reference (issue #2774)

OPENAPI_GEN := tools/openapi-gen
APISERVER_OPENAPI_JSON := website/static/openapi/apiserver/apiserver.openapi.json
APISERVER_REFERENCE_MD := website/docs/api/apiserver.md
APISERVER_INDEX_BEGIN := <!-- BEGIN-GENERATED-ENDPOINT-INDEX -->
APISERVER_INDEX_END := <!-- END-GENERATED-ENDPOINT-INDEX -->

.PHONY: generated-contract-check generated-contract-generate
generated-contract-check: config-schema-check api-docs-check agent-skill-check docs-generated-check docs-crd-check ## Check all generated public references without rewriting

# OpenAPI embeds the config schema: regenerate it before exporting API docs.
generated-contract-generate: config-schema-generate ## Regenerate OpenAPI, config contracts, and the public skill package in dependency order
	@$(MAKE) api-docs-generate
	@$(MAKE) agent-skill-sync
	@$(MAKE) model-catalog-generate docs-cli docs-config docs-crd

.PHONY: api-docs-openapi
api-docs-openapi: $(if $(CI),rust-ci,rust) ## Export committed apiserver OpenAPI JSON artifact from the route catalog
	@$(LOG_TARGET)
	@mkdir -p $(dir $(APISERVER_OPENAPI_JSON))
	@cd src/semantic-router && \
		CGO_ENABLED=1 \
		$(NATIVE_ENV) \
		go run ../../$(OPENAPI_GEN)/main.go -format json -o ../../$(APISERVER_OPENAPI_JSON)
	@echo "Wrote $(APISERVER_OPENAPI_JSON)"

.PHONY: api-docs-generate
api-docs-generate: api-docs-openapi ## Regenerate the apiserver reference endpoint index from the route catalog
	@$(LOG_TARGET)
	@cd src/semantic-router && \
		CGO_ENABLED=1 \
		$(NATIVE_ENV) \
		go run ../../$(OPENAPI_GEN)/main.go -format index -o /tmp/apiserver-endpoint-index.md
	@python3 tools/agent/scripts/embed_generated_index.py \
		--markdown "$(APISERVER_REFERENCE_MD)" \
		--index /tmp/apiserver-endpoint-index.md \
		--begin "$(APISERVER_INDEX_BEGIN)" \
		--end "$(APISERVER_INDEX_END)"

.PHONY: api-docs-check
api-docs-check: $(if $(CI),rust-ci,rust) ## Fail if committed api docs artifacts differ from generator output
	@$(LOG_TARGET)
	@TMPDIR_CHECK=$$(mktemp -d) && \
	trap 'rm -rf "$$TMPDIR_CHECK"' EXIT HUP INT TERM && \
	cp "$(APISERVER_REFERENCE_MD)" "$$TMPDIR_CHECK/apiserver.md" && \
	cd src/semantic-router && \
		CGO_ENABLED=1 \
		$(NATIVE_ENV) \
		go run ../../$(OPENAPI_GEN)/main.go -format json -o "$$TMPDIR_CHECK/apiserver.openapi.json" && \
		CGO_ENABLED=1 \
		$(NATIVE_ENV) \
		go run ../../$(OPENAPI_GEN)/main.go -format index -o "$$TMPDIR_CHECK/apiserver-endpoint-index.md" && \
	cd ../.. && \
	python3 tools/agent/scripts/embed_generated_index.py \
		--markdown "$$TMPDIR_CHECK/apiserver.md" \
		--index "$$TMPDIR_CHECK/apiserver-endpoint-index.md" \
		--begin "$(APISERVER_INDEX_BEGIN)" \
		--end "$(APISERVER_INDEX_END)" >/dev/null && \
	if ! diff -q "$$TMPDIR_CHECK/apiserver.openapi.json" "$(APISERVER_OPENAPI_JSON)" >/dev/null 2>&1; then \
		echo "ERROR: $(APISERVER_OPENAPI_JSON) is stale. Run 'make api-docs-openapi' and commit the result." >&2; \
		diff "$$TMPDIR_CHECK/apiserver.openapi.json" "$(APISERVER_OPENAPI_JSON)" | head -40 >&2; \
		exit 1; \
	fi && \
	if ! diff -q "$$TMPDIR_CHECK/apiserver.md" "$(APISERVER_REFERENCE_MD)" >/dev/null 2>&1; then \
		echo "ERROR: $(APISERVER_REFERENCE_MD) is stale. Run 'make api-docs-generate' and commit the result." >&2; \
		diff "$$TMPDIR_CHECK/apiserver.md" "$(APISERVER_REFERENCE_MD)" | head -40 >&2; \
		exit 1; \
	fi
	@echo "api-docs artifacts are up to date"
