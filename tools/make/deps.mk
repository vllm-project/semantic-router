# ====================== deps.mk ======================
# = External Binary Tool Dependencies Management      =
# ====================== deps.mk ======================

##@ Dependencies

# Project root directory
ROOT_DIR := $(shell pwd)

# Local tools binary directory - all external tools install here
TOOLS_BIN_DIR := $(ROOT_DIR)/tools/bin

# Export PATH so all subsequent commands can find local tools
export PATH := $(TOOLS_BIN_DIR):$(PATH)

# ====================== Tool Versions ======================
# Pin versions explicitly for reproducibility
FUNC_E_VERSION ?= 1.1.3

# ====================== Tool Binaries ======================
FUNC_E := $(TOOLS_BIN_DIR)/func-e

# ====================== Download Rules ======================

$(TOOLS_BIN_DIR):
	@mkdir -p $@

$(FUNC_E): | $(TOOLS_BIN_DIR)
	@echo "Downloading func-e $(FUNC_E_VERSION) to $(TOOLS_BIN_DIR)..."
	@curl -sSfL https://func-e.io/install.sh | bash -s -- -b $(TOOLS_BIN_DIR)
	@echo "func-e installed at $(FUNC_E)"

.PHONY: tools-clean
tools-clean: ## Remove all downloaded tool binaries
	@$(LOG_TARGET)
	rm -rf $(TOOLS_BIN_DIR)
	@echo "Cleaned $(TOOLS_BIN_DIR)"
