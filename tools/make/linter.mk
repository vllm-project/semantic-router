# =============================== linter.mk ==========================
# =  Everything For Project Linter, markdown, yaml, code spell etc.  =
# =============================== linter.mk ==========================

##@ Linter

markdown-lint: ## Lint all markdown files in the project
	@$(LOG_TARGET)
	markdownlint -c tools/linter/markdown/markdownlint.yaml "**/*.md" \
		--ignore node_modules \
		--ignore website/node_modules \
		--ignore dashboard/frontend/node_modules \
		--ignore models

markdown-lint-fix: ## Auto-fix markdown lint issues
	@$(LOG_TARGET)
	markdownlint -c tools/linter/markdown/markdownlint.yaml "**/*.md" \
		--ignore node_modules \
		--ignore website/node_modules \
		--ignore dashboard/frontend/node_modules \
		--ignore models \
		--fix

yaml-lint: ## Lint all YAML files in the project
	@$(LOG_TARGET)
	yamllint --config-file=tools/linter/yaml/.yamllint .

codespell: CODESPELL_SKIP := $(shell cat tools/linter/codespell/.codespell.skip | tr \\n ',')
codespell: ## Check for common misspellings in code and docs
	@$(LOG_TARGET)
	codespell --skip $(CODESPELL_SKIP) --ignore-words tools/linter/codespell/.codespell.ignorewords --check-filenames

shellcheck: ## Lint all shell scripts in the project
	@$(LOG_TARGET)
	@if ! command -v shellcheck >/dev/null 2>&1; then \
		echo "❌ Error: shellcheck is not installed"; \
		echo ""; \
		echo "To install shellcheck:"; \
		echo "  macOS:   brew install shellcheck"; \
		echo "  Ubuntu:  sudo apt-get install shellcheck"; \
		echo "  Fedora:  sudo dnf install shellcheck"; \
		echo ""; \
		echo "Or skip shellcheck in pre-commit by running:"; \
		echo "  SKIP=shellcheck pre-commit run --all-files"; \
		exit 1; \
	fi
	@shellcheck --rcfile=tools/linter/shellcheck/.shellcheckrc $(shell find . -type f -name "*.sh" -not -path "./node_modules/*" -not -path "./website/node_modules/*" -not -path "./dashboard/frontend/node_modules/*" -not -path "./models/*" -not -path "./.venv/*")