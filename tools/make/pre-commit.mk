# Copyright 2025 The vLLM Semantic Router Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

PRECOMMIT_CONTAINER := ghcr.io/vllm-project/semantic-router/precommit:latest

precommit-install:
	pip install pre-commit

precommit-check:
	@echo "Running pre-commit on all files..."
	pre-commit run --all-files

# Run pre-commit hooks in a Docker container,
# and you can exec container to run bash for debug.
# export PRECOMMIT_CONTAINER=ghcr.io/vllm-project/semantic-router/precommit:latest
# docker run --rm -it \
#     -v $(pwd):/app \
#     -w /app \
#     --name precommit-container ${PRECOMMIT_CONTAINER} \
#     bash
# and then, run `pre-commit install && pre-commit run --all-files` command
precommit-local:
	@if command -v docker > /dev/null 2>&1; then \
		CONTAINER_CMD=docker; \
	elif command -v podman > /dev/null 2>&1; then \
		CONTAINER_CMD=podman; \
	else \
		echo "Error: Neither docker nor podman is installed. Please install one of them."; \
		exit 1; \
	fi; \
	if ! $$CONTAINER_CMD image inspect ${PRECOMMIT_CONTAINER} > /dev/null 2>&1; then \
		echo "Image not found locally. Pulling..."; \
		$$CONTAINER_CMD pull ${PRECOMMIT_CONTAINER}; \
	else \
		echo "Image found locally. Skipping pull."; \
	fi; \
	$$CONTAINER_CMD run --rm \
	    -v $(shell pwd):/app \
	    -w /app \
	    --name precommit-container ${PRECOMMIT_CONTAINER} \
	    bash -c "source ~/.cargo/env && pre-commit install && pre-commit run --all-files"
