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

# Semantic Router Root Makefile Define.
# It is refer tools/make/*.mk as the sub-makefiles.

_run:
	@$(MAKE) --warn-undefined-variables \
		-f tools/make/common.mk \
		-f tools/make/envs.mk \
		-f tools/make/envoy.mk \
		-f tools/make/golang.mk \
		-f tools/make/rust.mk \
		-f tools/make/build-run-test.mk \
		-f tools/make/docs.mk \
		-f tools/make/linter.mk \
		-f tools/make/milvus.mk \
		-f tools/make/models.mk \
		-f tools/make/pre-commit.mk \
		$(MAKECMDGOALS)

.PHONY: _run

$(if $(MAKECMDGOALS),$(MAKECMDGOALS): %: _run)
