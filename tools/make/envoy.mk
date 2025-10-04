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

# ======== envoy.mk ========
# = Everything For envoy   =
# ======== envoy.mk ========

# Prepare Envoy
prepare-envoy:
	@$(LOG_TARGET)
	curl https://func-e.io/install.sh | sudo bash -s -- -b /usr/local/bin

# Run Envoy proxy
run-envoy:
	@$(LOG_TARGET)
	@echo "Checking for func-e..."
	@if ! command -v func-e >/dev/null 2>&1; then \
		echo "func-e not found, installing..."; \
		$(MAKE) prepare-envoy; \
	fi
	@echo "Starting Envoy..."
	func-e run --config-path config/envoy.yaml --component-log-level "ext_proc:trace,router:trace,http:trace"
