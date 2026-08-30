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
	"math"
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
	strategy       config.RoutingStrategy
	routingScope   config.RecipeName
}

// WithRoutingScope namespaces observability state for recipe-local decision
// names. It does not alter matching or the public DecisionResult.
func (e *DecisionEngine) WithRoutingScope(recipeName config.RecipeName) *DecisionEngine {
	if e != nil {
		e.routingScope = recipeName
	}
	return e
}

// NewDecisionEngine creates a new decision engine
func NewDecisionEngine(
	keywordRules []config.KeywordRule,
	embeddingRules []config.EmbeddingRule,
	categories []config.Category,
	decisions []config.Decision,
	strategy config.RoutingStrategy,
) *DecisionEngine {
	if strategy == "" {
		strategy = config.RoutingStrategyPriority
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
	KeywordRules       []string
	EmbeddingRules     []string
	DomainRules        []string
	FactCheckRules     []string // "needs_fact_check" or "no_fact_check_needed"
	UserFeedbackRules  []string // "need_clarification", "satisfied", "want_different", "wrong_answer"
	ReaskRules         []string // History-aware dissatisfaction signals from repeated user turns
	PreferenceRules    []string // Route preference names matched via external LLM
	LanguageRules      []string // Language codes: "en", "es", "zh", "fr", etc.
	ContextRules       []string // Context rule names matched (e.g. "low_token_count")
	StructureRules     []string // Structure rule names matched (e.g. "many_questions")
	ComplexityRules    []string // Complexity rules with difficulty level (e.g. "code_complexity:hard")
	ModalityRules      []string // Modality classification: "AR", "DIFFUSION", or "BOTH"
	AuthzRules         []string // Authz rule names matched for user-level routing (e.g. "premium_tier")
	JailbreakRules     []string // Jailbreak rule names matched (confidence >= threshold)
	PIIRules           []string // PII rule names matched (denied PII types detected)
	KBRules            []string // KB signal names matched from global.model_catalog.kbs bindings
	ConversationRules  []string // Conversation-shape signal names matched
	EventRules         []string // event rule names (event type, severity, temporal, action codes)
	MetadataRules      []string // untrusted request metadata rule names matched
	ClassifierRules    []string // generic classifier label names matched
	InputModalityRules []string // structural input-modality presence rule names matched
	ProjectionRules    []string // Derived routing outputs from routing.projections.mappings

	SignalConfidences map[string]float64 // "signalType:ruleName" → real score (0.0-1.0), e.g. {"embedding:ai": 0.88}. Defaults to 1.0 if missing
	SignalValues      map[string]float64 // raw numeric values exposed by signal evaluators
	SignalErrors      map[string]string  // signal evaluation errors keyed by "type:name"
}

// DecisionResult represents the result of decision evaluation
type DecisionResult struct {
	Decision        *config.Decision
	Confidence      float64
	MatchedRules    []string
	MatchedKeywords []string // The actual keywords that matched (not rule names)

	// ConfidenceScored reports whether every value that contributed to
	// Confidence came from a reported signal score. Signals that never
	// report a confidence (keyword, language, pii, ...), predicate gates,
	// and NOT guards all contribute the structural constant 1.0 instead;
	// that constant is not comparable with calibrated or similarity-based
	// scores, so selection only ranks by confidence within pools where
	// every competitor is scored.
	ConfidenceScored bool
	// CatchAll marks a decision with no conditions (omitted rules or an
	// empty AND). Catch-alls rank after signal-backed decisions wherever
	// confidence-based selection applies, regardless of scoring.
	CatchAll bool
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
		matched, confidence, scored, matchedRules := e.evaluateDecisionWithSignals(decision, signals)

		if matched {
			// Record decision match with confidence
			metrics.RecordDecisionMatch(config.RoutingDecisionKey(e.routingScope, decision.Name), confidence)

			results = append(results, DecisionResult{
				Decision:         decision,
				Confidence:       confidence,
				MatchedRules:     matchedRules,
				ConfidenceScored: scored,
				CatchAll:         isCatchAllRules(decision.Rules),
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
// scored reports whether every contribution to confidence came from a reported
// signal score rather than a structural constant; see DecisionResult.
func (e *DecisionEngine) evaluateDecisionWithSignals(
	decision *config.Decision,
	signals *SignalMatches,
) (matched bool, confidence float64, scored bool, matchedRules []string) {
	// Omitting rules is the YAML equivalent of a DSL route without WHEN. Keep
	// this root-only contract explicit instead of relying on the zero value to
	// fall through to OR semantics.
	if decision.Rules.IsEmpty() {
		return true, 0, true, nil
	}
	return e.evalNode(decision.Rules, signals)
}

// isCatchAllRules reports whether a decision claims no conditions at all:
// omitted rules or the empty-AND authoring form.
func isCatchAllRules(rules config.RuleCombination) bool {
	if rules.IsEmpty() {
		return true
	}
	return !rules.IsLeaf() && strings.ToUpper(rules.Operator) == "AND" && len(rules.Conditions) == 0
}

// evalNode recursively evaluates a RuleNode (boolean expression tree) against signal matches.
// Leaf nodes check whether a specific named signal is present.
// Composite nodes apply AND / OR / NOT logic over their children.
func (e *DecisionEngine) evalNode(
	node config.RuleNode,
	signals *SignalMatches,
) (matched bool, confidence float64, scored bool, matchedRules []string) {
	if node.IsLeaf() {
		return e.evalLeaf(node, signals)
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
	node config.RuleNode,
	signals *SignalMatches,
) (matched bool, confidence float64, scored bool, matchedRules []string) {
	normalizedType := strings.ToLower(strings.TrimSpace(node.Type))

	matched, supported := e.matchesSignalType(normalizedType, node.Name, signals)
	if !supported {
		return false, 0, false, nil
	}
	if node.Predicate != nil {
		return evaluatePredicateLeaf(node, normalizedType, signals)
	}
	if !matched {
		return false, 0, false, nil
	}

	confidence, scored = signalConfidence(signals.SignalConfidences, normalizedType, node.Name)
	return true, confidence, scored, []string{formatMatchedRule(node)}
}

func evaluatePredicateLeaf(
	node config.RuleNode,
	normalizedType string,
	signals *SignalMatches,
) (bool, float64, bool, []string) {
	// Predicates are boolean gates: they rank with the structural constant
	// 1.0, which is not a reported score, so they never count as scored.
	value, available := signalPredicateValue(signals, normalizedType, node.Name, node.Label)
	if available {
		if numericPredicateMatches(value, node.Predicate) {
			return true, 1.0, false, []string{formatMatchedRule(node)}
		}
		return false, 0, false, nil
	}
	errorKey := fmt.Sprintf("%s:%s", normalizedType, node.Name)
	_, failed := signals.SignalErrors[errorKey]
	if failed && strings.EqualFold(strings.TrimSpace(node.OnError), "match") {
		return true, 1.0, false, []string{formatMatchedRule(node)}
	}
	return false, 0, false, nil
}

func formatMatchedRule(node config.RuleNode) string {
	rule := fmt.Sprintf("%s:%s", node.Type, node.Name)
	if node.Label != "" {
		rule += ":" + node.Label
	}
	return rule
}

func signalPredicateValue(
	signals *SignalMatches,
	signalType string,
	name string,
	label string,
) (float64, bool) {
	key := fmt.Sprintf("%s:%s", signalType, name)
	if label != "" {
		key += ":" + label
	}
	if signals.SignalValues != nil {
		if value, ok := signals.SignalValues[key]; ok {
			return value, true
		}
	}
	if signals.SignalConfidences != nil {
		value, ok := signals.SignalConfidences[key]
		return value, ok
	}
	return 0, false
}

func numericPredicateMatches(value float64, predicate *config.NumericPredicate) bool {
	if predicate == nil {
		return true
	}
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return false
	}
	if predicate.GT != nil && value <= *predicate.GT {
		return false
	}
	if predicate.GTE != nil && value < *predicate.GTE {
		return false
	}
	if predicate.LT != nil && value >= *predicate.LT {
		return false
	}
	if predicate.LTE != nil && value > *predicate.LTE {
		return false
	}
	return true
}

func (e *DecisionEngine) matchesSignalType(
	normalizedType string,
	name string,
	signals *SignalMatches,
) (matched bool, supported bool) {
	if normalizedType == "domain" {
		return e.matchesDomainCondition(name, signals.DomainRules), true
	}
	if normalizedType == config.SignalTypeClassifier {
		// Classifier conditions are predicate-only and configuration validation
		// guarantees that the named classifier exists.
		return false, true
	}

	rules, ok := resolveSignalRules(normalizedType, signals)
	if !ok {
		return false, false
	}
	return slices.Contains(rules, name), true
}

func resolveSignalRules(
	signalType string,
	signals *SignalMatches,
) ([]string, bool) {
	if rules, ok := resolvePrimarySignalRules(signalType, signals); ok {
		return rules, true
	}
	return resolvePolicySignalRules(signalType, signals)
}

func resolvePrimarySignalRules(
	signalType string,
	signals *SignalMatches,
) ([]string, bool) {
	switch signalType {
	case config.SignalTypeKeyword:
		return signals.KeywordRules, true
	case config.SignalTypeEmbedding:
		return signals.EmbeddingRules, true
	case config.SignalTypeFactCheck:
		return signals.FactCheckRules, true
	case config.SignalTypeUserFeedback:
		return signals.UserFeedbackRules, true
	case config.SignalTypeReask:
		return signals.ReaskRules, true
	case config.SignalTypePreference:
		return signals.PreferenceRules, true
	case config.SignalTypeLanguage:
		return signals.LanguageRules, true
	case config.SignalTypeContext:
		return signals.ContextRules, true
	case config.SignalTypeStructure:
		return signals.StructureRules, true
	case config.SignalTypeComplexity:
		return signals.ComplexityRules, true
	default:
		return nil, false
	}
}

func resolvePolicySignalRules(
	signalType string,
	signals *SignalMatches,
) ([]string, bool) {
	switch signalType {
	case config.SignalTypeModality:
		return signals.ModalityRules, true
	case config.SignalTypeAuthz:
		return signals.AuthzRules, true
	case config.SignalTypeJailbreak:
		return signals.JailbreakRules, true
	case config.SignalTypePII:
		return signals.PIIRules, true
	case config.SignalTypeKB:
		return signals.KBRules, true
	case config.SignalTypeConversation:
		return signals.ConversationRules, true
	case config.SignalTypeEvent:
		return signals.EventRules, true
	case config.SignalTypeMetadata:
		return signals.MetadataRules, true
	case config.SignalTypeInputModality:
		return signals.InputModalityRules, true
	case config.SignalTypeProjection:
		return signals.ProjectionRules, true
	default:
		return nil, false
	}
}

// signalConfidence returns the reported score for a signal and whether one
// was reported at all. Signals that report nothing rank with the structural
// default 1.0, which selection must not compare against reported scores.
func signalConfidence(confidences map[string]float64, signalType string, name string) (float64, bool) {
	if confidences == nil {
		return 1.0, false
	}

	signalKey := fmt.Sprintf("%s:%s", signalType, name)
	if score, ok := confidences[signalKey]; ok {
		return score, true
	}
	return 1.0, false
}

// evalAND returns true only when every child matches.
// An empty conjunction acts as a catch-all/default route with zero confidence,
// so it can serve as a fallback without outranking signal-backed decisions when
// confidence-based selection is enabled.
func (e *DecisionEngine) evalAND(
	children []config.RuleNode,
	signals *SignalMatches,
) (matched bool, confidence float64, scored bool, matchedRules []string) {
	if len(children) == 0 {
		return true, 0, true, nil
	}
	totalConf := 0.0
	scored = true
	for _, child := range children {
		m, c, s, r := e.evalNode(child, signals)
		if !m {
			return false, 0, false, nil
		}
		totalConf += c
		scored = scored && s
		matchedRules = append(matchedRules, r...)
	}
	return true, totalConf / float64(len(children)), scored, matchedRules
}

// evalOR returns true when at least one child matches; returns the best-confidence match.
func (e *DecisionEngine) evalOR(
	children []config.RuleNode,
	signals *SignalMatches,
) (matched bool, confidence float64, scored bool, matchedRules []string) {
	bestConf := 0.0
	bestScored := false
	var bestRules []string
	for _, child := range children {
		m, c, s, r := e.evalNode(child, signals)
		if m && (!matched || c > bestConf) {
			matched = true
			bestConf = c
			bestScored = s
			bestRules = r
		}
	}
	if matched {
		return true, bestConf, bestScored, bestRules
	}
	return false, 0, false, nil
}

// evalNOT is a strictly unary operator: it negates the result of its single child.
// Configuration errors (0 or 2+ children) are treated as non-matching.
func (e *DecisionEngine) evalNOT(
	children []config.RuleNode,
	signals *SignalMatches,
) (matched bool, confidence float64, scored bool, matchedRules []string) {
	if len(children) != 1 {
		logging.Warnf("NOT operator requires exactly 1 child, got %d — treating as non-match", len(children))
		return false, 0, false, nil
	}
	m, c, s, r := e.evalNode(children[0], signals)
	if !m {
		// Child did not match → NOT matches. The 1.0 is a structural
		// constant (absence of evidence, not strength of evidence), so a
		// matching NOT guard is never scored.
		return true, 1.0, false, r
	}
	// Child matched → NOT does not match.
	return false, c, s, r
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
	comparable := e.comparableConfidencePools(results, useTieredSelection)
	sort.Slice(results, func(i, j int) bool {
		return e.decisionResultLess(results[i], results[j], useTieredSelection, comparable)
	})

	return &results[0]
}

// comparableConfidencePools reports, per competing pool, whether confidence
// ordering is meaningful there: every non-catch-all member's confidence must
// be evidence-scored. Signals that report no confidence rank with the
// structural constant 1.0, which would otherwise outrank any honestly
// reported score, so a pool containing such a member falls back to the
// documented priority ordering instead. Pools are tiers under tiered
// selection and the whole result set otherwise.
func (e *DecisionEngine) comparableConfidencePools(
	results []DecisionResult,
	useTieredSelection bool,
) map[int]bool {
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

func (e *DecisionEngine) confidencePool(result DecisionResult, useTieredSelection bool) int {
	if useTieredSelection {
		return result.Decision.Tier
	}
	return 0
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
		if left.Decision.Tier != right.Decision.Tier {
			return left.Decision.Tier < right.Decision.Tier
		}
		if left.CatchAll != right.CatchAll {
			return right.CatchAll
		}
		if comparable[e.confidencePool(left, true)] && left.Confidence != right.Confidence {
			return left.Confidence > right.Confidence
		}
		if left.Decision.Priority != right.Decision.Priority {
			return left.Decision.Priority > right.Decision.Priority
		}
		return left.Decision.Name < right.Decision.Name
	}

	if e.strategy == config.RoutingStrategyConfidence {
		if left.CatchAll != right.CatchAll {
			return right.CatchAll
		}
		if comparable[0] && left.Confidence != right.Confidence {
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
	if comparable[0] && left.Confidence != right.Confidence {
		return left.Confidence > right.Confidence
	}
	return left.Decision.Name < right.Decision.Name
}
