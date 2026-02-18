# ========================== docs.mk ==========================
# = Everything For Docs,include API Docs and Docs Website     =
# ========================== docs.mk ==========================

##@ Docs

docs-install: ## Install documentation website dependencies
	@$(LOG_TARGET)
	cd website && npm install

docs-dev: docs-install ## Start documentation website in dev mode
	@$(LOG_TARGET)
	cd website && npm start

docs-dev-zh: docs-install ## Start documentation website in dev mode
	@$(LOG_TARGET)
	cd website && npm run start:zh

docs-build: docs-install ## Build static documentation website
	@$(LOG_TARGET)
	cd website && npm run build

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

##@ CRD Documentation

CRD_REF_DOCS_VERSION ?= latest
CRD_REF_DOCS := $(shell command -v crd-ref-docs 2> /dev/null)

.PHONY: install-crd-ref-docs
install-crd-ref-docs: ## Install crd-ref-docs tool
	@$(LOG_TARGET)
	@if [ -z "$(CRD_REF_DOCS)" ]; then \
		echo "Installing crd-ref-docs..."; \
		go install github.com/elastic/crd-ref-docs@$(CRD_REF_DOCS_VERSION); \
	else \
		echo "crd-ref-docs is already installed at $(CRD_REF_DOCS)"; \
	fi

.PHONY: docs-crd
docs-crd: install-crd-ref-docs markdown-lint-fix ## Generate CRD API reference documentation
	@$(LOG_TARGET)
	@echo "Generating CRD documentation from Go API types..."
	@if [ -d "src/semantic-router/pkg/apis/vllm.ai/v1alpha1" ]; then \
		crd-ref-docs \
			--source-path=./src/semantic-router/pkg/apis/vllm.ai/v1alpha1 \
			--config=.crd-ref-docs.yaml \
			--renderer=markdown \
			--output-path=./website/docs/api/crd-reference.md; \
		echo "✅ CRD documentation generated at website/docs/api/crd-reference.md"; \
	else \
		echo "⚠️  API directory not found, generating from CRD YAML files..."; \
		crd-ref-docs \
			--source-path=./deploy/kubernetes/crds \
			--renderer=markdown \
			--output-path=./website/docs/api/crd-reference.md; \
		echo "✅ CRD documentation generated from YAML at website/docs/api/crd-reference.md"; \
	fi
	@echo "📝 Adding Docusaurus frontmatter..."
	@if ! grep -q "^---" website/docs/api/crd-reference.md; then \
		echo "---" > website/docs/api/crd-reference.md.tmp; \
		echo "sidebar_position: 3" >> website/docs/api/crd-reference.md.tmp; \
		echo "title: CRD API Reference" >> website/docs/api/crd-reference.md.tmp; \
		echo "description: Kubernetes Custom Resource Definitions (CRDs) API reference for vLLM Semantic Router" >> website/docs/api/crd-reference.md.tmp; \
		echo "---" >> website/docs/api/crd-reference.md.tmp; \
		echo "" >> website/docs/api/crd-reference.md.tmp; \
		cat website/docs/api/crd-reference.md >> website/docs/api/crd-reference.md.tmp; \
		mv website/docs/api/crd-reference.md.tmp website/docs/api/crd-reference.md; \
		echo "✅ Frontmatter added"; \
	else \
		echo "✅ Frontmatter already exists"; \
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
docs-all: docs-crd docs-build ## Generate all documentation (CRD + website)
	@$(LOG_TARGET)
	@echo "✅ All documentation generated successfully"

