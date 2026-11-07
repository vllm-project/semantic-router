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
	"fmt"
	"math"
)

// parseVerifierUsage accepts older servers that omit usage, but never invents
// zero consumption for them. Counts include every internal verifier sample.
func parseVerifierUsage(raw json.RawMessage) (TokenUsage, error) {
	usage := TokenUsage{Unreported: true}
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return usage, nil
	}
	if err := json.Unmarshal(raw, &usage); err != nil {
		return TokenUsage{Unreported: true}, fmt.Errorf("invalid verifier usage")
	}
	if usage.PromptTokens < 0 || usage.CompletionTokens < 0 || usage.TotalTokens < 0 ||
		usage.TotalTokens-usage.PromptTokens != usage.CompletionTokens ||
		usage.CachedInputTokens < 0 || usage.CacheWriteTokens < 0 ||
		usage.CachedInputTokens > usage.PromptTokens || usage.CacheWriteTokens > usage.PromptTokens-usage.CachedInputTokens {
		return TokenUsage{Unreported: true}, fmt.Errorf("inconsistent verifier usage")
	}
	return usage, nil
}

func accumulateVerifierUsage(total *TokenUsage, usage TokenUsage) error {
	if total.PromptTokens > math.MaxInt64-usage.PromptTokens ||
		total.CompletionTokens > math.MaxInt64-usage.CompletionTokens ||
		total.TotalTokens > math.MaxInt64-usage.TotalTokens ||
		total.CachedInputTokens > math.MaxInt64-usage.CachedInputTokens ||
		total.CacheWriteTokens > math.MaxInt64-usage.CacheWriteTokens {
		total.Unreported = true
		return fmt.Errorf("verifier usage exceeds accounting range")
	}
	*total = total.Add(&ModelResponse{Usage: usage})
	return nil
}
