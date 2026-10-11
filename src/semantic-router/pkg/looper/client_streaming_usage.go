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
	usage, _ := parseStreamingUsageWithPresence(body)
	return usage
}

func parseStreamingUsageWithPresence(body []byte) (TokenUsage, UsagePresence) {
	usage := TokenUsage{Unreported: true}
	var presence UsagePresence
	for _, line := range bytes.Split(body, []byte("\n")) {
		data, ok := sseDataPayload(line)
		if !ok {
			continue
		}
		var chunk struct {
			Usage json.RawMessage `json:"usage"`
		}
		if json.Unmarshal(data, &chunk) != nil || len(chunk.Usage) == 0 || bytes.Equal(bytes.TrimSpace(chunk.Usage), []byte("null")) {
			continue
		}
		usage, presence = parseResponseUsageWithPresence(data)
	}
	return usage, presence
}

// A present null, negative, fractional, or nonnumeric field is not token evidence.
// Numeric zero is explicitly reported usage.
func parseResponseUsagePresence(body []byte) UsagePresence {
	var response struct {
		Usage map[string]json.RawMessage `json:"usage"`
	}
	if json.Unmarshal(body, &response) != nil {
		return UsagePresence{}
	}
	valid := func(key string) bool {
		var value *int64
		return json.Unmarshal(response.Usage[key], &value) == nil && value != nil && *value >= 0
	}
	return UsagePresence{PromptTokens: valid("prompt_tokens"), CompletionTokens: valid("completion_tokens"), TotalTokens: valid("total_tokens")}
}

// Retain upstream cache accounting while preserving individually reported counts.
func parseResponseUsageWithPresence(body []byte) (TokenUsage, UsagePresence) {
	usage := parseResponseUsage(body)
	presence := parseResponseUsagePresence(body)
	var fields struct {
		Usage map[string]json.RawMessage `json:"usage"`
	}
	if json.Unmarshal(body, &fields) == nil {
		if presence.PromptTokens {
			_ = json.Unmarshal(fields.Usage["prompt_tokens"], &usage.PromptTokens)
		}
		if presence.CompletionTokens {
			_ = json.Unmarshal(fields.Usage["completion_tokens"], &usage.CompletionTokens)
		}
		if presence.TotalTokens {
			_ = json.Unmarshal(fields.Usage["total_tokens"], &usage.TotalTokens)
		}
	}
	if !presence.PromptTokens || !presence.CompletionTokens || !presence.TotalTokens {
		usage.Unreported = true
	}
	return usage, presence
}
