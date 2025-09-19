# =============================== linter.mk ==========================
# =  Everything For Project Linter, markdown, yaml, code spell etc.  =
# =============================== linter.mk ==========================

docs-lint: docs-install
	@$(LOG_TARGET)
	cd website && npm run lint

docs-lint-fix: docs-install
	@$(LOG_TARGET)
	cd website && npm run lint:fix

markdown-lint:
	@$(LOG_TARGET)
	markdownlint -c tools/linter/markdown/markdownlint.yaml "**/*.md" --ignore node_modules --ignore website/node_modules

markdown-lint-fix:
	@$(LOG_TARGET)
	markdownlint -c tools/linter/markdown/markdownlint.yaml "**/*.md" --ignore node_modules --ignore website/node_modules --fix

yaml-lint:
	@$(LOG_TARGET)
	yamllint --config-file=tools/linter/yaml/.yamllint .

codespell: CODESPELL_SKIP := $(shell cat tools/linter/codespell/.codespell.skip | tr \\n ',')
codespell:
	@$(LOG_TARGET)
	codespell --skip $(CODESPELL_SKIP) --ignore-words tools/linter/codespell/.codespell.ignorewords --check-filenames
