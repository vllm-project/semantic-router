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
	ReaskRules        []string // History-aware dissatisfaction signals from repeated user turns
	PreferenceRules   []string // Route preference names matched via external LLM
	LanguageRules     []string // Language codes: "en", "es", "zh", "fr", etc.
	ContextRules      []string // Context rule names matched (e.g. "low_token_count")
	StructureRules    []string // Structure rule names matched (e.g. "many_questions")
	ComplexityRules   []string // Complexity rules with difficulty level (e.g. "code_complexity:hard")
	ModalityRules     []string // Modality classification: "AR", "DIFFUSION", or "BOTH"
	AuthzRules        []string // Authz rule names matched for user-level routing (e.g. "premium_tier")
	JailbreakRules    []string // Jailbreak rule names matched (confidence >= threshold)
	PIIRules          []string // PII rule names matched (denied PII types detected)
	KBRules           []string // KB signal names matched from global.model_catalog.kbs bindings
	ProjectionRules   []string // Derived routing outputs from routing.projections.mappings

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

	matched, supported := e.matchesSignalType(normalizedType, name, signals)
	if !supported {
		return false, 0, nil
	}
	if !matched {
		return false, 0, nil
	}

	confidence = signalConfidence(signals.SignalConfidences, normalizedType, name)
	return true, confidence, []string{fmt.Sprintf("%s:%s", typ, name)}
}

func (e *DecisionEngine) matchesSignalType(
	normalizedType string,
	name string,
	signals *SignalMatches,
) (matched bool, supported bool) {
	if normalizedType == "domain" {
		return e.matchesDomainCondition(name, signals.DomainRules), true
	}

	ruleSets := map[string][]string{
		"keyword":       signals.KeywordRules,
		"embedding":     signals.EmbeddingRules,
		"fact_check":    signals.FactCheckRules,
		"user_feedback": signals.UserFeedbackRules,
		"reask":         signals.ReaskRules,
		"preference":    signals.PreferenceRules,
		"language":      signals.LanguageRules,
		"context":       signals.ContextRules,
		"structure":     signals.StructureRules,
		"complexity":    signals.ComplexityRules,
		"modality":      signals.ModalityRules,
		"authz":         signals.AuthzRules,
		"jailbreak":     signals.JailbreakRules,
		"pii":           signals.PIIRules,
		"kb":            signals.KBRules,
		"projection":    signals.ProjectionRules,
	}

	rules, ok := ruleSets[normalizedType]
	if !ok {
		return false, false
	}
	return slices.Contains(rules, name), true
}

func signalConfidence(confidences map[string]float64, signalType string, name string) float64 {
	if confidences == nil {
		return 1.0
	}

	signalKey := fmt.Sprintf("%s:%s", signalType, name)
	if score, ok := confidences[signalKey]; ok && score > 0 {
		return score
	}
	return 1.0
}

// evalAND returns true only when every child matches.
// An empty conjunction acts as a catch-all/default route with zero confidence,
// so it can serve as a fallback without outranking signal-backed decisions when
// confidence-based selection is enabled.
func (e *DecisionEngine) evalAND(
	children []config.RuleNode,
	signals *SignalMatches,
) (matched bool, confidence float64, matchedRules []string) {
	if len(children) == 0 {
		return true, 0, nil
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

	useTieredSelection := e.useTieredSelection(results)
	sort.Slice(results, func(i, j int) bool {
		return e.decisionResultLess(results[i], results[j], useTieredSelection)
	})

	return &results[0]
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
) bool {
	if useTieredSelection {
		if left.Decision.Tier != right.Decision.Tier {
			return left.Decision.Tier < right.Decision.Tier
		}
		if left.Confidence != right.Confidence {
			return left.Confidence > right.Confidence
		}
		if left.Decision.Priority != right.Decision.Priority {
			return left.Decision.Priority > right.Decision.Priority
		}
		return left.Decision.Name < right.Decision.Name
	}

	if e.strategy == "confidence" {
		if left.Confidence != right.Confidence {
			return left.Confidence > right.Confidence
		}
		if left.Decision.Priority != right.Decision.Priority {
			return left.Decision.Priority > right.Decision.Priority
		}
		return left.Decision.Name < right.Decision.Name
	}

	if left.Decision.Priority != right.Decision.Priority {
		return left.Decision.Priority > right.Decision.Priority
	}
	if left.Confidence != right.Confidence {
		return left.Confidence > right.Confidence
	}
	return left.Decision.Name < right.Decision.Name
}
