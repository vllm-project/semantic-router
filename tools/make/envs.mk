# ====================== envs.mk ======================
# = Environment Makefile, refer for other makefile    =
# ====================== envs.mk ======================

# Container runtime (docker or podman)
CONTAINER_RUNTIME ?= docker

# vLLM env var
VLLM_ENDPOINT ?=

# Config file path with default
CONFIG_FILE ?= config/config.yaml

# Tag is the tag to use for build and push image targets.
TAG ?= $(REV)
