/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package looper

import (
	"bytes"
	"encoding/json"
)

// parseStreamingUsage extracts token usage from an SSE stream. OpenAI-compatible
// backends report usage in a trailing chunk (only when the request set
// stream_options.include_usage), so the last non-null usage block wins. Returns
// zero usage when none is present.
func parseStreamingUsage(body []byte) TokenUsage {
	var usage TokenUsage
	for _, line := range bytes.Split(body, []byte("\n")) {
		lineStr := string(line)
		if len(lineStr) <= 6 || lineStr[:6] != "data: " {
			continue
		}
		data := lineStr[6:]
		if data == "[DONE]" {
			continue
		}

		var chunk struct {
			Usage *TokenUsage `json:"usage"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}
		if chunk.Usage != nil {
			usage = *chunk.Usage
		}
	}
	return usage
}
