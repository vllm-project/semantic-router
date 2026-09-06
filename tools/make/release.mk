##@ Release

.PHONY: built-in-model-snapshot release release-check

built-in-model-snapshot: ## Freeze config/recipes/built-in/latest into vMAJOR.MINOR before tagging
	@test -n "$(RELEASE_VERSION)" || { \
		echo "RELEASE_VERSION is required. Example: make built-in-model-snapshot RELEASE_VERSION=X.Y.Z"; \
		exit 1; \
	}
	@python3 tools/release/snapshot_model_catalog.py --version "$(RELEASE_VERSION)"
	@python3 tools/release/sync_model_catalog.py
	@python3 tools/release/snapshot_model_catalog.py --check --version "$(RELEASE_VERSION)"
	@python3 tools/release/sync_model_catalog.py --check

release-check: ## Validate the local release version contract; set RELEASE_VERSION to check a tag version
	@python3 tools/release/check_version_contract.py $(if $(RELEASE_VERSION),--version "$(RELEASE_VERSION)")

release: ## Prepare the local vllm-sr release commit, stable tag, and next dev bump
	@test -n "$(RELEASE_VERSION)" || { \
		echo "RELEASE_VERSION is required. Example: make release RELEASE_VERSION=0.3.0 NEXT_VERSION=0.4.0"; \
		exit 1; \
	}
	@src/vllm-sr/scripts/release.sh "$(RELEASE_VERSION)" "$(NEXT_VERSION)"
