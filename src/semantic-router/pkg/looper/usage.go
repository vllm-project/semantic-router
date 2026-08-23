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
	"math"
)

// TokenUsage is Looper's compact aggregate presentation of neutral input,
// output, and total usage. It never drives quota settlement; BackendInvoker's
// per-dispatch response terminals are the accounting authority.
type TokenUsage struct {
	PromptTokens     int64 `json:"prompt_tokens"`
	CompletionTokens int64 `json:"completion_tokens"`
	TotalTokens      int64 `json:"total_tokens"`
	invalid          bool
}

// NewActualTokenUsage constructs usage from provider-reported token counts.
// Invalid counts produce an unknown value instead of silently becoming zero.
func NewActualTokenUsage(promptTokens, completionTokens, totalTokens int64) TokenUsage {
	return validatedTokenUsage(TokenUsage{
		PromptTokens: promptTokens, CompletionTokens: completionTokens, TotalTokens: totalTokens,
	})
}

// UnknownTokenUsage returns a non-authoritative usage value. Callers that
// persist TokenUsage outside this package must preserve Authoritative().
func UnknownTokenUsage() TokenUsage { return unknownTokenUsage() }

// Authoritative reports whether all token counts came from valid provider
// usage rather than an absent, malformed, or overflowing observation.
func (u TokenUsage) Authoritative() bool { return u.isValid() }

// Add returns u with the usage of the given responses added to it. It is
// nil-safe: nil responses contribute nothing, so callers can accumulate across
// rounds or skip failed calls without guarding. The receiver is not mutated.
func (u TokenUsage) Add(resps ...*ModelResponse) TokenUsage {
	if !u.isValid() {
		return unknownTokenUsage()
	}
	for _, resp := range resps {
		if resp == nil {
			continue
		}
		usage := validatedTokenUsage(resp.Usage)
		if !usage.isValid() {
			return unknownTokenUsage()
		}
		u = mergeTokenUsage(u, usage)
		if !u.isValid() {
			return u
		}
	}
	return u
}

func validatedTokenUsage(usage TokenUsage) TokenUsage {
	if !usage.isValid() {
		return unknownTokenUsage()
	}
	return usage
}

func (u TokenUsage) isValid() bool {
	return !u.invalid && u.PromptTokens >= 0 && u.CompletionTokens >= 0 && u.TotalTokens >= 0
}

func unknownTokenUsage() TokenUsage {
	return TokenUsage{invalid: true}
}

func mergeTokenUsage(first, second TokenUsage) TokenUsage {
	if !first.isValid() || !second.isValid() {
		return unknownTokenUsage()
	}
	if first.PromptTokens > math.MaxInt64-second.PromptTokens ||
		first.CompletionTokens > math.MaxInt64-second.CompletionTokens ||
		first.TotalTokens > math.MaxInt64-second.TotalTokens {
		return unknownTokenUsage()
	}
	return TokenUsage{
		PromptTokens:     first.PromptTokens + second.PromptTokens,
		CompletionTokens: first.CompletionTokens + second.CompletionTokens,
		TotalTokens:      first.TotalTokens + second.TotalTokens,
	}
}

// SumUsage sums the per-call usage of the given responses. nil responses are
// skipped. The total is computed from each response's reported prompt and
// completion tokens; TotalTokens is taken from the backend rather than
// recomputed so it matches the upstream accounting.
func SumUsage(resps ...*ModelResponse) TokenUsage {
	return TokenUsage{}.Add(resps...)
}
