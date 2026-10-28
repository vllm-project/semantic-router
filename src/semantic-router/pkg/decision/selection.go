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

func (e *DecisionEngine) selectBestDecision(results []DecisionResult) (*DecisionResult, *RankingTrace) {
	if len(results) == 0 {
		return nil, nil
	}
	useTieredSelection := e.useTieredSelection(results)
	comparable, fallbacks := comparableConfidencePools(results, useTieredSelection)
	var runnerUp *DecisionResult
	if len(results) > 1 {
		sort.Slice(results, func(i, j int) bool {
			return e.decisionResultLess(results[i], results[j], useTieredSelection, comparable)
		})
		runnerUp = &results[1]
	}
	return &results[0], e.rankingTrace(results[0], runnerUp, useTieredSelection, comparable, fallbacks, len(results))
}

// comparableConfidencePools reports the pools whose confidences rank against
// each other. A pool qualifies when every member that is not a catch-all
// reported a score and every one of those scores is the same kind, since a
// classifier probability and a vector similarity are different quantities.
// It also reports, for every pool it disqualified, the decision that made it
// incomparable and the reason, which the ranking trace repeats to whoever asks
// why priority decided.
func comparableConfidencePools(results []DecisionResult, useTieredSelection bool) (map[int]bool, map[int]string) {
	pools := make(map[int]bool)
	fallbacks := make(map[int]string)
	kinds := make(map[int]config.ScoreKind)
	for _, result := range results {
		key := 0
		if useTieredSelection {
			key = result.Decision.Tier
		}
		comparable, seen := pools[key]
		if !seen {
			comparable = true
		}
		if !result.CatchAll && comparable {
			switch kind, ranked := kinds[key]; {
			case !result.ConfidenceScored:
				comparable = false
				fallbacks[key] = result.Decision.Name + " reported no comparable score"
			case !ranked:
				kinds[key] = result.ScoreKind
			case kind != result.ScoreKind:
				comparable = false
				fallbacks[key] = result.Decision.Name + " reported a " + string(result.ScoreKind) + " against a " + string(kind)
			}
		}
		pools[key] = comparable
	}
	return pools, fallbacks
}

func (e *DecisionEngine) useTieredSelection(results []DecisionResult) bool {
	for _, result := range results {
		if result.Decision != nil && result.Decision.Tier > 0 {
			return true
		}
	}
	return false
}

// decisionResultLess orders two matched decisions. Tier is the hard
// precedence boundary, a catch-all always ranks after a real match, and
// routing.strategy then decides between policy ordering and evidence, the
// same way inside a tier as without one.
func (e *DecisionEngine) decisionResultLess(
	left DecisionResult,
	right DecisionResult,
	useTieredSelection bool,
	comparable map[int]bool,
) bool {
	pool := 0
	if useTieredSelection {
		if left.Decision.Tier != right.Decision.Tier {
			return left.Decision.Tier < right.Decision.Tier
		}
		pool = left.Decision.Tier
	}
	if left.CatchAll != right.CatchAll {
		return right.CatchAll
	}
	rankedByConfidence := comparable[pool] && left.Confidence != right.Confidence
	samePriority := left.Decision.Priority == right.Decision.Priority
	if e.strategy == config.RoutingStrategyConfidence {
		if rankedByConfidence {
			return left.Confidence > right.Confidence
		}
		if !samePriority {
			return left.Decision.Priority > right.Decision.Priority
		}
		return left.Decision.Name < right.Decision.Name
	}
	if !samePriority {
		return left.Decision.Priority > right.Decision.Priority
	}
	if rankedByConfidence {
		return left.Confidence > right.Confidence
	}
	return left.Decision.Name < right.Decision.Name
}
