##@ Release

.PHONY: release

release: ## Prepare the local vllm-sr release commit, stable tag, and next dev bump
	@test -n "$(RELEASE_VERSION)" || { \
		echo "RELEASE_VERSION is required. Example: make release RELEASE_VERSION=0.3.0 NEXT_VERSION=0.4.0"; \
		exit 1; \
	}
	@src/vllm-sr/scripts/release.sh "$(RELEASE_VERSION)" "$(NEXT_VERSION)"
