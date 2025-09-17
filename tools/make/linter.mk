# =============================== linter.mk ==========================
# = Everything For Project Linter, markdown, yaml, code spell etc.   =
# =============================== linter.mk ==========================

docs-lint:
	@$(LOG_TARGET)
	cd website && npm run lint

docs-lint-fix:
	@$(LOG_TARGET)
	cd website && npm run lint:fix

markdown-lint:
	@$(LOG_TARGET)
	markdownlint -c markdownlint.yaml "**/*.md" --ignore node_modules --ignore website/node_modules

markdown-lint-fix:
	@$(LOG_TARGET)
	markdownlint -c markdownlint.yaml "**/*.md" --ignore node_modules --ignore website/node_modules --fix

yaml-lint:
	@$(LOG_TARGET)
	yamllint --config-file=.yamllint .
