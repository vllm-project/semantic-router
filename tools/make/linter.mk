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

# License header checking and fixing
license-check:
	@$(LOG_TARGET)
	@echo "Checking license headers..."
	@if ! command -v license-eye >/dev/null 2>&1; then \
		echo "Installing license-eye..."; \
		go install github.com/apache/skywalking-eyes/cmd/license-eye@latest; \
	fi
	license-eye header check

license-fix:
	@$(LOG_TARGET)
	@echo "Fixing license headers..."
	@if ! command -v license-eye >/dev/null 2>&1; then \
		echo "Installing license-eye..."; \
		go install github.com/apache/skywalking-eyes/cmd/license-eye@latest; \
	fi
	license-eye header fix
