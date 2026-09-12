# ====================== common.mk ======================
# = Common function or variables for other makefiles    =
# ====================== common.mk ======================

##@ Common

# Turn off .INTERMEDIATE file removal by marking all files as
# .SECONDARY.  .INTERMEDIATE file removal is a space-saving hack from
# a time when drives were small; on modern computers with plenty of
# storage, it causes nothing but headaches.
#
# https://news.ycombinator.com/item?id=16486331
.SECONDARY:

# Variables Define
DATETIME = $(shell date +"%Y%m%d%H%M%S")

# REV is the short git sha of latest commit.
REV=$(shell git rev-parse --short HEAD)

# The router links all native providers in one process. Keep the build and
# runtime search paths independent of each recipe's working directory, while
# preserving caller-supplied vendor/OpenVINO paths and linker options.
NATIVE_LIBRARY_DIRS := $(addsuffix /target/release,$(addprefix $(CURDIR)/,candle-binding onnx-binding ml-binding nlp-binding))
native_empty :=
native_space := $(native_empty) $(native_empty)
NATIVE_LIBRARY_PATH = $(subst $(native_space),:,$(NATIVE_LIBRARY_DIRS))
NATIVE_LDFLAGS = $(addprefix -L,$(NATIVE_LIBRARY_DIRS))
NATIVE_ENV = LD_LIBRARY_PATH="$(NATIVE_LIBRARY_PATH)$${LD_LIBRARY_PATH:+:$${LD_LIBRARY_PATH}}" CGO_LDFLAGS="$(NATIVE_LDFLAGS)$${CGO_LDFLAGS:+ $${CGO_LDFLAGS}}"

# Function Define

# logging Output Function
# Log normal info
LOG_TARGET = echo "\033[0;32m==================> Running $@ ============> ... \033[0m"

# Log debugging info
define log
echo "\033[36m==================>$1\033[0m"
endef

# Log error info
define errorLog
echo "\033[0;31m==================>$1\033[0m"
endef

## help: Show this help info.
.PHONY: help
help: ## Show help info.
	@echo "\033[1;3;34mVllm semantic-router: Intelligent Mixture-of-Models Router for Efficient LLM Inference.\033[0m\n"
	@echo "Usage:\n  make \033[36m<Target>\033[0m \033[36m<Option>\033[0m\n\nTargets:"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
