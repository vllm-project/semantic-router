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
	ContextRules      []string // Context rule names matched (e.g. "low_token_count")
	ComplexityRules   []string // Complexity rules with difficulty level (e.g. "code_complexity:hard")
	ModalityRules     []string // Modality classification: "AR", "DIFFUSION", or "BOTH"
	AuthzRules        []string // Authz rule names matched for user-level routing (e.g. "premium_tier")

	SignalConfidences map[string]float64 // "signalType:ruleName" → real score (0.0-1.0), e.g. {"embedding:ai": 0.88}. Defaults to 1.0 if missing
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

// evaluateDecisionWithSignals evaluates a single decision's rule tree with all signals.
func (e *DecisionEngine) evaluateDecisionWithSignals(
	decision *config.Decision,
	signals *SignalMatches,
) (matched bool, confidence float64, matchedRules []string) {
	return e.evalNode(decision.Rules, signals)
}

// evalNode recursively evaluates a RuleNode (boolean expression tree) against signal matches.
// Leaf nodes check whether a specific named signal is present.
// Composite nodes apply AND / OR / NOT logic over their children.
func (e *DecisionEngine) evalNode(
	node config.RuleNode,
	signals *SignalMatches,
) (matched bool, confidence float64, matchedRules []string) {
	if node.IsLeaf() {
		return e.evalLeaf(node.Type, node.Name, signals)
	}

	switch strings.ToUpper(node.Operator) {
	case "AND":
		return e.evalAND(node.Conditions, signals)
	case "NOT":
		return e.evalNOT(node.Conditions, signals)
	default: // OR
		return e.evalOR(node.Conditions, signals)
	}
}

// evalLeaf evaluates a single signal condition (leaf node).
func (e *DecisionEngine) evalLeaf(
	typ, name string,
	signals *SignalMatches,
) (matched bool, confidence float64, matchedRules []string) {
	normalizedType := strings.ToLower(strings.TrimSpace(typ))

	switch normalizedType {
	case "keyword":
		matched = slices.Contains(signals.KeywordRules, name)
	case "embedding":
		matched = slices.Contains(signals.EmbeddingRules, name)
	case "domain":
		matched = e.matchesDomainCondition(name, signals.DomainRules)
	case "fact_check":
		matched = slices.Contains(signals.FactCheckRules, name)
	case "user_feedback":
		matched = slices.Contains(signals.UserFeedbackRules, name)
	case "preference":
		matched = slices.Contains(signals.PreferenceRules, name)
	case "language":
		matched = slices.Contains(signals.LanguageRules, name)
	case "context":
		matched = slices.Contains(signals.ContextRules, name)
	case "complexity":
		matched = slices.Contains(signals.ComplexityRules, name)
	case "modality":
		matched = slices.Contains(signals.ModalityRules, name)
	case "authz":
		matched = slices.Contains(signals.AuthzRules, name)
	default:
		return false, 0, nil
	}

	if !matched {
		return false, 0, nil
	}

	// Use real confidence score if available (e.g., embedding similarity = 0.88),
	// otherwise fall back to 1.0 for backward compatibility.
	signalKey := fmt.Sprintf("%s:%s", normalizedType, name)
	if signals.SignalConfidences != nil {
		if score, ok := signals.SignalConfidences[signalKey]; ok && score > 0 {
			confidence = score
		} else {
			confidence = 1.0
		}
	} else {
		confidence = 1.0
	}

	return true, confidence, []string{fmt.Sprintf("%s:%s", typ, name)}
}

// evalAND returns true only when every child matches; confidence is the average.
func (e *DecisionEngine) evalAND(
	children []config.RuleNode,
	signals *SignalMatches,
) (matched bool, confidence float64, matchedRules []string) {
	if len(children) == 0 {
		return false, 0, nil
	}
	totalConf := 0.0
	for _, child := range children {
		m, c, r := e.evalNode(child, signals)
		if !m {
			return false, 0, nil
		}
		totalConf += c
		matchedRules = append(matchedRules, r...)
	}
	return true, totalConf / float64(len(children)), matchedRules
}

// evalOR returns true when at least one child matches; returns the best-confidence match.
func (e *DecisionEngine) evalOR(
	children []config.RuleNode,
	signals *SignalMatches,
) (matched bool, confidence float64, matchedRules []string) {
	bestConf := 0.0
	var bestRules []string
	for _, child := range children {
		m, c, r := e.evalNode(child, signals)
		if m {
			matched = true
			if c > bestConf {
				bestConf = c
				bestRules = r
			}
		}
	}
	if matched {
		return true, bestConf, bestRules
	}
	return false, 0, nil
}

// evalNOT is a strictly unary operator: it negates the result of its single child.
// Configuration errors (0 or 2+ children) are treated as non-matching.
func (e *DecisionEngine) evalNOT(
	children []config.RuleNode,
	signals *SignalMatches,
) (matched bool, confidence float64, matchedRules []string) {
	if len(children) != 1 {
		logging.Warnf("NOT operator requires exactly 1 child, got %d — treating as non-match", len(children))
		return false, 0, nil
	}
	m, c, r := e.evalNode(children[0], signals)
	if !m {
		// Child did not match → NOT matches with full certainty.
		return true, 1.0, r
	}
	// Child matched → NOT does not match.
	return false, c, r
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
