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
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
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

// SignalMatches contains all matched signals for decision evaluation
type SignalMatches struct {
	KeywordRules      []string
	EmbeddingRules    []string
	DomainRules       []string
	FactCheckRules    []string // "needs_fact_check" or "no_fact_check_needed"
	UserFeedbackRules []string // "need_clarification", "satisfied", "want_different", "wrong_answer"
	PreferenceRules   []string // Route preference names matched via external LLM
	LanguageRules     []string // Language codes: "en", "es", "zh", "fr", etc.
	LatencyRules      []string // Latency rule names that matched based on model TPOT
	ContextRules      []string // Context rule names matched (e.g. "low_token_count")
}

// DecisionResult represents the result of decision evaluation
type DecisionResult struct {
	Decision        *config.Decision
	Confidence      float64
	MatchedRules    []string
	MatchedKeywords []string // The actual keywords that matched (not rule names)
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
	// Call EvaluateDecisionsWithSignals with empty fact_check rules for backward compatibility
	return e.EvaluateDecisionsWithSignals(&SignalMatches{
		KeywordRules:   matchedKeywordRules,
		EmbeddingRules: matchedEmbeddingRules,
		DomainRules:    matchedDomainRules,
		FactCheckRules: nil,
	})
}

// EvaluateDecisionsWithSignals evaluates all decisions using SignalMatches
// This is the new method that supports all signal types including fact_check
func (e *DecisionEngine) EvaluateDecisionsWithSignals(signals *SignalMatches) (*DecisionResult, error) {
	// Record decision evaluation start time
	start := time.Now()
	defer func() {
		latencySeconds := time.Since(start).Seconds()
		metrics.RecordDecisionEvaluation(latencySeconds)
	}()

	if len(e.decisions) == 0 {
		return nil, fmt.Errorf("no decisions configured")
	}

	var results []DecisionResult

	// Evaluate each decision
	for i := range e.decisions {
		decision := &e.decisions[i]
		matched, confidence, matchedRules := e.evaluateDecisionWithSignals(decision, signals)

		if matched {
			// Record decision match with confidence
			metrics.RecordDecisionMatch(decision.Name, confidence)

			results = append(results, DecisionResult{
				Decision:     decision,
				Confidence:   confidence,
				MatchedRules: matchedRules,
			})
		}
	}

	if len(results) == 0 {
		logging.Infof("No decision matched")
		return nil, nil
	}

	// Select best decision based on strategy
	return e.selectBestDecision(results), nil
}

// evaluateDecisionWithSignals evaluates a single decision's rule combination with all signals
func (e *DecisionEngine) evaluateDecisionWithSignals(
	decision *config.Decision,
	signals *SignalMatches,
) (matched bool, confidence float64, matchedRules []string) {
	return e.evaluateRuleCombinationWithSignals(decision.Rules, signals)
}

// evaluateRuleCombinationWithSignals evaluates a rule combination with all signal types
func (e *DecisionEngine) evaluateRuleCombinationWithSignals(
	rules config.RuleCombination,
	signals *SignalMatches,
) (matched bool, confidence float64, matchedRules []string) {
	if len(rules.Conditions) == 0 {
		return false, 0, nil
	}

	matchedCount := 0
	totalConfidence := 0.0
	var allMatchedRules []string

	for _, condition := range rules.Conditions {
		conditionMatched := false

		// Normalize condition type to lowercase for case-insensitive matching
		// All signal types are normalized to match constants and switch cases
		normalizedType := strings.ToLower(strings.TrimSpace(condition.Type))

		switch normalizedType {
		case "keyword":
			conditionMatched = slices.Contains(signals.KeywordRules, condition.Name)
		case "embedding":
			conditionMatched = slices.Contains(signals.EmbeddingRules, condition.Name)
		case "domain":
			// Domain matching: check if the detected domain matches the category
			// A match occurs if:
			// 1. The detected domain equals the category name, OR
			// 2. The detected domain is in the category's mmlu_categories list
			conditionMatched = e.matchesDomainCondition(condition.Name, signals.DomainRules)
		case "fact_check":
			conditionMatched = slices.Contains(signals.FactCheckRules, condition.Name)
		case "user_feedback":
			conditionMatched = slices.Contains(signals.UserFeedbackRules, condition.Name)
		case "preference":
			conditionMatched = slices.Contains(signals.PreferenceRules, condition.Name)
		case "language":
			conditionMatched = slices.Contains(signals.LanguageRules, condition.Name)
		case "latency":
			conditionMatched = slices.Contains(signals.LatencyRules, condition.Name)
		case "context":
			conditionMatched = slices.Contains(signals.ContextRules, condition.Name)
		default:
			continue
		}

		if conditionMatched {
			matchedCount++
			totalConfidence += 1.0 // Each matched condition contributes 1.0 to confidence
			allMatchedRules = append(allMatchedRules, fmt.Sprintf("%s:%s", condition.Type, condition.Name))
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

// matchesDomainCondition checks if any of the detected domains match the given category name
// A match occurs if:
// 1. The detected domain equals the category name directly, OR
// 2. The detected domain is in the category's mmlu_categories list
func (e *DecisionEngine) matchesDomainCondition(categoryName string, detectedDomains []string) bool {
	// Direct match: detected domain equals the category name
	if slices.Contains(detectedDomains, categoryName) {
		return true
	}

	// Check if any detected domain is in the category's mmlu_categories
	for _, cat := range e.categories {
		if cat.Name == categoryName {
			for _, detectedDomain := range detectedDomains {
				if slices.Contains(cat.MMLUCategories, detectedDomain) {
					return true
				}
			}
			break // Found the category, no need to continue
		}
	}
	return false
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
