# ====================== envs.mk ======================
# = Environment Makefile, refer for other makefile    =
# ====================== envs.mk ======================

# CI environment flag (set by CI/CD systems like GitHub Actions)
# Can be overridden: CI=true make build
CI ?=

# Container runtime (docker or podman)
CONTAINER_RUNTIME ?= docker

# vLLM env var
VLLM_ENDPOINT ?=

# Config file path with default
CONFIG_FILE ?= config/config.yaml

# Tag is the tag to use for build and push image targets.
TAG ?= $(REV)
