/*
Copyright 2026 vLLM Semantic Router.

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

package decision

import (
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func (e *DecisionEngine) selectBestDecision(results []DecisionResult) *DecisionResult {
	if len(results) == 0 {
		return nil
	}
	if len(results) == 1 {
		return &results[0]
	}

	useTieredSelection := e.useTieredSelection(results)
	comparable := comparableConfidencePools(results, useTieredSelection)
	sort.Slice(results, func(i, j int) bool {
		return e.decisionResultLess(results[i], results[j], useTieredSelection, comparable)
	})
	return &results[0]
}

func comparableConfidencePools(results []DecisionResult, useTieredSelection bool) map[int]bool {
	pools := make(map[int]bool)
	for _, result := range results {
		key := 0
		if useTieredSelection {
			key = result.Decision.Tier
		}
		comparable, seen := pools[key]
		if !seen {
			comparable = true
		}
		if !result.CatchAll && !result.ConfidenceScored {
			comparable = false
		}
		pools[key] = comparable
	}
	return pools
}

func (e *DecisionEngine) useTieredSelection(results []DecisionResult) bool {
	for _, result := range results {
		if result.Decision != nil && result.Decision.Tier > 0 {
			return true
		}
	}
	return false
}

func (e *DecisionEngine) decisionResultLess(
	left DecisionResult,
	right DecisionResult,
	useTieredSelection bool,
	comparable map[int]bool,
) bool {
	if useTieredSelection {
		return tieredDecisionResultLess(left, right, comparable[left.Decision.Tier])
	}
	if e.strategy == config.RoutingStrategyConfidence {
		return confidenceDecisionResultLess(left, right, comparable[0])
	}
	return priorityDecisionResultLess(left, right, comparable[0])
}

func tieredDecisionResultLess(left, right DecisionResult, comparable bool) bool {
	if left.Decision.Tier != right.Decision.Tier {
		return left.Decision.Tier < right.Decision.Tier
	}
	if left.CatchAll != right.CatchAll {
		return right.CatchAll
	}
	if comparable && left.Confidence != right.Confidence {
		return left.Confidence > right.Confidence
	}
	if left.Decision.Priority != right.Decision.Priority {
		return left.Decision.Priority > right.Decision.Priority
	}
	return left.Decision.Name < right.Decision.Name
}

func confidenceDecisionResultLess(left, right DecisionResult, comparable bool) bool {
	if left.CatchAll != right.CatchAll {
		return right.CatchAll
	}
	if comparable && left.Confidence != right.Confidence {
		return left.Confidence > right.Confidence
	}
	if left.Decision.Priority != right.Decision.Priority {
		return left.Decision.Priority > right.Decision.Priority
	}
	return left.Decision.Name < right.Decision.Name
}

func priorityDecisionResultLess(left, right DecisionResult, comparable bool) bool {
	if left.Decision.Priority != right.Decision.Priority {
		return left.Decision.Priority > right.Decision.Priority
	}
	if comparable && left.Confidence != right.Confidence {
		return left.Confidence > right.Confidence
	}
	return left.Decision.Name < right.Decision.Name
}
