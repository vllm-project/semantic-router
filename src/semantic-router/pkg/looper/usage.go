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

// TokenUsage holds OpenAI-compatible token counts. For multi-model looper
// algorithms it represents the aggregate across every model call made during a
// single execution (panel + judge + synthesis, rounds, candidates, etc.).
type TokenUsage struct {
	PromptTokens     int64 `json:"prompt_tokens"`
	CompletionTokens int64 `json:"completion_tokens"`
	TotalTokens      int64 `json:"total_tokens"`
}

// UsagePresence records which token fields were actually present in the
// provider response. A zero token count is valid evidence, so callers must not
// infer presence from TokenUsage values alone.
type UsagePresence struct {
	PromptTokens     bool
	CompletionTokens bool
	TotalTokens      bool
}

// Any reports whether the provider included at least one usage field.
func (p UsagePresence) Any() bool {
	return p.PromptTokens || p.CompletionTokens || p.TotalTokens
}

// UsageKnown reports whether a response has enough accounting data to charge
// a provider total without falling back to its reservation estimate.
func (r *ModelResponse) UsageKnown() bool {
	if r == nil {
		return false
	}
	if r.UsagePresent.TotalTokens ||
		(r.UsagePresent.PromptTokens && r.UsagePresent.CompletionTokens) {
		return true
	}
	// Responses assembled by existing unit tests predate UsagePresent. A
	// non-zero usage value is still treated as known for compatibility; an all
	// zero value remains unknown because zero is also the omitted-usage value.
	if r.UsagePresent.Any() {
		return false
	}
	return r.Usage != (TokenUsage{})
}

// Add returns u with the usage of the given responses added to it. It is
// nil-safe: nil responses contribute nothing, so callers can accumulate across
// rounds or skip failed calls without guarding. The receiver is not mutated.
func (u TokenUsage) Add(resps ...*ModelResponse) TokenUsage {
	for _, resp := range resps {
		if resp == nil {
			continue
		}
		u.PromptTokens += resp.Usage.PromptTokens
		u.CompletionTokens += resp.Usage.CompletionTokens
		u.TotalTokens += resp.Usage.TotalTokens
	}
	return u
}

// SumUsage sums the per-call usage of the given responses. nil responses are
// skipped. The total is computed from each response's reported prompt and
// completion tokens; TotalTokens is taken from the backend rather than
// recomputed so it matches the upstream accounting.
func SumUsage(resps ...*ModelResponse) TokenUsage {
	return TokenUsage{}.Add(resps...)
}

// Map renders the usage as the OpenAI-compatible block embedded in a
// chat.completion response body. This is the single seam every looper uses in
// place of the legacy hardcoded {0,0,0} literal.
func (u TokenUsage) Map() map[string]interface{} {
	return map[string]interface{}{
		"prompt_tokens":     u.PromptTokens,
		"completion_tokens": u.CompletionTokens,
		"total_tokens":      u.TotalTokens,
	}
}
