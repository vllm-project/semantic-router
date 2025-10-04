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

# ========================== docs.mk ==========================
# = Everything For Docs,include API Docs and Docs Website     =
# ========================== docs.mk ==========================

# Documentation targets
docs-install:
	@$(LOG_TARGET)
	cd website && npm install

docs-dev: docs-install
	@$(LOG_TARGET)
	cd website && npm start

docs-build: docs-install
	@$(LOG_TARGET)
	cd website && npm run build

docs-serve: docs-build
	@$(LOG_TARGET)
	cd website && npm run serve

docs-clean:
	@$(LOG_TARGET)
	cd website && npm run clear
