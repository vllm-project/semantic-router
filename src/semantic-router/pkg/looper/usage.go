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

import "encoding/json"

// TokenUsage holds OpenAI-compatible token counts. For multi-model looper
// algorithms it represents the aggregate across every model call made during a
// single execution (panel + judge + synthesis, rounds, candidates, etc.).
type TokenUsage struct {
	PromptTokens      int64 `json:"prompt_tokens"`
	CompletionTokens  int64 `json:"completion_tokens"`
	TotalTokens       int64 `json:"total_tokens"`
	CachedInputTokens int64 `json:"cached_input_tokens,omitempty"`
	CacheWriteTokens  int64 `json:"cache_write_tokens,omitempty"`
	Unreported        bool  `json:"-"`
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
		u.CachedInputTokens += resp.Usage.CachedInputTokens
		u.CacheWriteTokens += resp.Usage.CacheWriteTokens
		u.Unreported = u.Unreported || resp.Usage.Unreported
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
	values := map[string]interface{}{
		"prompt_tokens":     u.PromptTokens,
		"completion_tokens": u.CompletionTokens,
		"total_tokens":      u.TotalTokens,
	}
	if u.CachedInputTokens > 0 {
		values["prompt_tokens_details"] = map[string]int64{"cached_tokens": u.CachedInputTokens}
	}
	if u.CacheWriteTokens > 0 {
		values["cache_write_tokens"] = u.CacheWriteTokens
	}
	return values
}

// Complete distinguishes reported, internally consistent usage from absent or
// partial accounting. Prompt tokens include both cache read and write buckets.
func (u TokenUsage) Complete() bool {
	return !u.Unreported && u.PromptTokens >= 0 && u.CompletionTokens >= 0 &&
		u.TotalTokens > 0 && u.TotalTokens-u.PromptTokens == u.CompletionTokens &&
		u.CachedInputTokens >= 0 && u.CacheWriteTokens >= 0 &&
		u.CachedInputTokens <= u.PromptTokens && u.CacheWriteTokens <= u.PromptTokens-u.CachedInputTokens
}

// UnmarshalJSON retains cache buckets that the SDK's base usage struct can lose.
func (u *TokenUsage) UnmarshalJSON(data []byte) error {
	var raw struct {
		Prompt     *int64 `json:"prompt_tokens"`
		Completion *int64 `json:"completion_tokens"`
		Total      *int64 `json:"total_tokens"`
		Cached     *int64 `json:"cached_input_tokens"`
		Write      *int64 `json:"cache_write_tokens"`
		Creation   *int64 `json:"cache_creation_input_tokens"`
		Details    struct {
			Cached   *int64 `json:"cached_tokens"`
			Write    *int64 `json:"cache_write_tokens"`
			Creation *int64 `json:"cache_creation_tokens"`
			Created  *int64 `json:"created_cache_tokens"`
		} `json:"prompt_tokens_details"`
	}
	*u = TokenUsage{Unreported: true}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	if raw.Prompt == nil || raw.Completion == nil || raw.Total == nil {
		return nil
	}
	u.PromptTokens, u.CompletionTokens, u.TotalTokens = *raw.Prompt, *raw.Completion, *raw.Total
	u.Unreported = false
	var cachedSeen, writeSeen bool
	for _, value := range []*int64{raw.Cached, raw.Details.Cached} {
		if value != nil {
			if cachedSeen && u.CachedInputTokens != *value {
				u.Unreported = true
			}
			u.CachedInputTokens = *value
			cachedSeen = true
		}
	}
	for _, value := range []*int64{raw.Write, raw.Creation, raw.Details.Write, raw.Details.Creation, raw.Details.Created} {
		if value != nil {
			if writeSeen && u.CacheWriteTokens != *value {
				u.Unreported = true
			}
			u.CacheWriteTokens = *value
			writeSeen = true
		}
	}
	return nil
}

func parseResponseUsage(body []byte) TokenUsage {
	var response struct {
		Usage *TokenUsage `json:"usage"`
	}
	if err := json.Unmarshal(body, &response); err != nil || response.Usage == nil {
		return TokenUsage{Unreported: true}
	}
	return *response.Usage
}
