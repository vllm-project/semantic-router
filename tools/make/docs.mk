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

