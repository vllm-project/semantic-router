##@ Supply Chain Security

SECURITY_DIR := tools/security
AST_SCANNER := $(SECURITY_DIR)/ast_security_scanner.py
REGEX_SCANNER := $(SECURITY_DIR)/scan_malicious_code.py

security-scan: ## Run full AST + regex supply chain security scan
	@echo "=== AST Supply Chain Security Scan ==="
	@python3 $(AST_SCANNER) scan . --fail-on HIGH
	@echo ""
	@echo "=== Regex Supply Chain Security Scan ==="
	@python3 $(REGEX_SCANNER) . --fail-on HIGH

security-scan-diff: ## Scan PR diff for supply chain attacks (AST-based)
	@echo "=== AST PR Diff Security Scan ==="
	@BASE=$${AGENT_BASE_REF:-$${GITHUB_BASE_REF:-main}}; \
	python3 $(AST_SCANNER) diff "$$BASE" --fail-on HIGH

security-scan-ci: ## CI gate: AST scan (for GitHub Actions)
	@echo "=== CI Security Gate ==="
	@python3 $(AST_SCANNER) scan . --fail-on HIGH

.PHONY: security-scan security-scan-diff security-scan-ci
