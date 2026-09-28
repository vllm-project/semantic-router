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

// sseDataPayload parses an SSE data field. The single space after the colon is
// optional in the SSE grammar, and OpenAI-compatible backends emit both forms.
func sseDataPayload(line []byte) ([]byte, bool) {
	line = bytes.TrimSuffix(line, []byte("\r"))
	if !bytes.HasPrefix(line, []byte("data:")) {
		return nil, false
	}
	payload := line[len("data:"):]
	if len(payload) > 0 && payload[0] == ' ' {
		payload = payload[1:]
	}
	return payload, true
}

// parseStreamingUsage extracts token usage from an SSE stream. OpenAI-compatible
// backends report usage in a trailing chunk (only when the request set
// stream_options.include_usage), so the last non-null usage block wins. Returns
// unreported usage when none is present.
func parseStreamingUsage(body []byte) TokenUsage {
	usage := TokenUsage{Unreported: true}
	for _, line := range bytes.Split(body, []byte("\n")) {
		data, ok := sseDataPayload(line)
		if !ok {
			continue
		}
		data = bytes.TrimSpace(data)
		if bytes.Equal(data, []byte("[DONE]")) {
			continue
		}

		var chunk struct {
			Usage json.RawMessage `json:"usage"`
		}
		if err := json.Unmarshal(data, &chunk); err != nil {
			continue
		}
		if len(chunk.Usage) > 0 && !bytes.Equal(bytes.TrimSpace(chunk.Usage), []byte("null")) {
			usage = parseResponseUsage(data)
		}
	}
	return usage
}
