#!/usr/bin/env bash
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

set -euo pipefail

CONFIG_FILE_PATH=${CONFIG_FILE:-/app/config/config.yaml}

if [[ ! -f "$CONFIG_FILE_PATH" ]]; then
  echo "[entrypoint] Config file not found at $CONFIG_FILE_PATH" >&2
  exit 1
fi

echo "[entrypoint] Starting semantic-router with config: $CONFIG_FILE_PATH"
exec /app/extproc-server --config "$CONFIG_FILE_PATH"
