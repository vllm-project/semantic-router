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

// ComputeBudget declares the resource ceiling for one Looper execution. It is
// the static, request-scoped policy; BudgetLedger tracks live consumption
// against it. A nil ComputeBudget (the default for every decision that does
// not configure algorithm.budget) means unlimited — every consumer of this
// type must treat nil as "no policy configured," not as a zero-value budget
// that blocks all work.
//
// ComputeBudget intentionally tracks only tokens, cost, and wall-time. It
// does not cap the number of upstream calls a request makes — that is a
// separate, non-overridable safety mechanism (see issue #1456) which is
// expected to compose with this budget at the same enforcement points
// (BudgetLedger.Exhausted checks), not be duplicated by it.
//
// Cost accounting is a conservative estimate, not an exact bill: it is
// computed from TokenUsage (prompt/completion token counts only) via
// modelpricing.Cost, which treats the full prompt token count as
// standard-rate input. Looper's ModelResponse does not currently carry the
// cached-input/cache-write token split a provider may report, so a request
// that actually benefited from prompt caching will show a higher estimated
// cost here than its real bill. This can only make MaxEstimatedCost trip
// earlier than the true cost would justify, never later.
type ComputeBudget struct {
	// MaxPromptTokens caps cumulative prompt tokens across every model call
	// in this execution. Zero means unlimited.
	MaxPromptTokens int64

	// MaxCompletionTokens caps cumulative completion tokens. Zero means
	// unlimited.
	MaxCompletionTokens int64

	// MaxTotalTokens caps cumulative prompt+completion tokens. Zero means
	// unlimited.
	MaxTotalTokens int64

	// MaxEstimatedCost caps cumulative estimated cost, in Currency. Zero
	// means unlimited.
	MaxEstimatedCost float64

	// Currency is the unit MaxEstimatedCost and the ledger's tracked cost are
	// denominated in (e.g. "USD"). Informational only; no cross-currency
	// conversion is performed.
	Currency string

	// MaxWallTimeMs caps the wall-clock duration of the full execution
	// (every model call plus algorithm overhead), enforced via
	// context.WithTimeout around the whole Looper.Execute call. Zero means
	// unlimited.
	MaxWallTimeMs int64
}

// IsZero reports whether b declares no limits at all, i.e. behaves
// identically to a nil budget. Nil-safe.
func (b *ComputeBudget) IsZero() bool {
	if b == nil {
		return true
	}
	return b.MaxPromptTokens == 0 &&
		b.MaxCompletionTokens == 0 &&
		b.MaxTotalTokens == 0 &&
		b.MaxEstimatedCost == 0 &&
		b.MaxWallTimeMs == 0
}
