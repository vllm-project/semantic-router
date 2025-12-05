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

package decision

import (
	"fmt"
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// DecisionEngine evaluates routing decisions based on rule combinations
type DecisionEngine struct {
	keywordRules   []config.KeywordRule
	embeddingRules []config.EmbeddingRule
	categories     []config.Category
	decisions      []config.Decision
	strategy       string
}

// NewDecisionEngine creates a new decision engine
func NewDecisionEngine(
	keywordRules []config.KeywordRule,
	embeddingRules []config.EmbeddingRule,
	categories []config.Category,
	decisions []config.Decision,
	strategy string,
) *DecisionEngine {
	if strategy == "" {
		strategy = "priority" // default strategy
	}
	return &DecisionEngine{
		keywordRules:   keywordRules,
		embeddingRules: embeddingRules,
		categories:     categories,
		decisions:      decisions,
		strategy:       strategy,
	}
}

// DecisionResult represents the result of decision evaluation
type DecisionResult struct {
	Decision     *config.Decision
	Confidence   float64
	MatchedRules []string
}

// EvaluateDecisions evaluates all decisions and returns the best match based on strategy
// matchedKeywordRules: list of matched keyword rule names
// matchedEmbeddingRules: list of matched embedding rule names
// matchedDomainRules: list of matched domain rule names (category names)
func (e *DecisionEngine) EvaluateDecisions(
	matchedKeywordRules []string,
	matchedEmbeddingRules []string,
	matchedDomainRules []string,
) (*DecisionResult, error) {
	if len(e.decisions) == 0 {
		return nil, fmt.Errorf("no decisions configured")
	}

	var results []DecisionResult

	// Evaluate each decision
	for i := range e.decisions {
		decision := &e.decisions[i]
		matched, confidence, matchedRules := e.evaluateDecision(
			decision,
			matchedKeywordRules,
			matchedEmbeddingRules,
			matchedDomainRules,
		)

		if matched {
			results = append(results, DecisionResult{
				Decision:     decision,
				Confidence:   confidence,
				MatchedRules: matchedRules,
			})
		}
	}

	if len(results) == 0 {
		return nil, fmt.Errorf("no decision matched")
	}

	// Select best decision based on strategy
	return e.selectBestDecision(results), nil
}

// evaluateDecision evaluates a single decision's rule combination
func (e *DecisionEngine) evaluateDecision(
	decision *config.Decision,
	matchedKeywordRules []string,
	matchedEmbeddingRules []string,
	matchedDomainRules []string,
) (matched bool, confidence float64, matchedRules []string) {
	return e.evaluateRuleCombination(
		decision.Rules,
		matchedKeywordRules,
		matchedEmbeddingRules,
		matchedDomainRules,
	)
}

// evaluateRuleCombination evaluates a rule combination with AND/OR logic
func (e *DecisionEngine) evaluateRuleCombination(
	rules config.RuleCombination,
	matchedKeywordRules []string,
	matchedEmbeddingRules []string,
	matchedDomainRules []string,
) (matched bool, confidence float64, matchedRules []string) {
	if len(rules.Conditions) == 0 {
		return false, 0, nil
	}

	matchedCount := 0
	totalConfidence := 0.0
	var allMatchedRules []string

	for _, condition := range rules.Conditions {
		conditionMatched := false
		var matchedList []string

		switch condition.Type {
		case "keyword":
			matchedList = matchedKeywordRules
		case "embedding":
			matchedList = matchedEmbeddingRules
		case "domain":
			matchedList = matchedDomainRules
		default:
			continue
		}

		// Check if the condition's rule name is in the matched list
		for _, ruleName := range matchedList {
			if ruleName == condition.Name {
				conditionMatched = true
				allMatchedRules = append(allMatchedRules, fmt.Sprintf("%s:%s", condition.Type, condition.Name))
				break
			}
		}

		if conditionMatched {
			matchedCount++
			totalConfidence += 1.0 // Each matched condition contributes 1.0 to confidence
		}
	}

	// Calculate final match result based on operator
	if rules.Operator == "AND" {
		matched = matchedCount == len(rules.Conditions)
	} else { // OR
		matched = matchedCount > 0
	}

	// Calculate confidence as ratio of matched conditions
	if len(rules.Conditions) > 0 {
		confidence = totalConfidence / float64(len(rules.Conditions))
	}

	return matched, confidence, allMatchedRules
}

// selectBestDecision selects the best decision based on the configured strategy
func (e *DecisionEngine) selectBestDecision(results []DecisionResult) *DecisionResult {
	if len(results) == 0 {
		return nil
	}

	if len(results) == 1 {
		return &results[0]
	}

	// Sort based on strategy
	if e.strategy == "confidence" {
		// Sort by confidence (descending)
		sort.Slice(results, func(i, j int) bool {
			return results[i].Confidence > results[j].Confidence
		})
	} else {
		// Default: priority strategy
		// Sort by priority (descending)
		sort.Slice(results, func(i, j int) bool {
			return results[i].Decision.Priority > results[j].Decision.Priority
		})
	}

	return &results[0]
}
