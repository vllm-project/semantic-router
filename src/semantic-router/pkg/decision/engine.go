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

	SignalConfidences  map[string]float64 // "signalType:ruleName" → real score (0.0-1.0), e.g. {"embedding:ai": 0.88}. Defaults to 1.0 if missing
	SignalValues       map[string]float64 // raw numeric values exposed by signal evaluators
	SignalErrors       map[string]string  // signal evaluation errors keyed by "type:name"
	SignalErrorMatches map[string]bool
}

type evaluationState uint8

const (
	evaluationFalse evaluationState = iota
	evaluationTrue
	evaluationUnknown
)

type nodeEvaluation struct {
	state        evaluationState
	confidence   float64
	scored       bool
	matchedRules []string
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
	start := time.Now()
	defer func() {
		metrics.RecordDecisionEvaluation(time.Since(start).Seconds())
	}()

	if len(e.decisions) == 0 {
		return nil, fmt.Errorf("no decisions configured")
	}

	var results []DecisionResult

	for i := range e.decisions {
		decision := &e.decisions[i]
		evaluation := e.evaluateDecisionWithSignals(decision, signals, false)
		if evaluation.state == evaluationUnknown {
			evaluation = e.evaluateDecisionWithSignals(decision, signals, true)
			if evaluation.state != evaluationTrue {
				logging.Debugf("Decision %q unresolved by signal errors %v", decision.Name, signals.SignalErrors)
			}
		}
		if evaluation.state == evaluationTrue {
			metrics.RecordDecisionMatch(config.RoutingDecisionKey(e.routingScope, decision.Name), evaluation.confidence)
			results = append(results, DecisionResult{
				Decision:         decision,
				Confidence:       evaluation.confidence,
				MatchedRules:     evaluation.matchedRules,
				ConfidenceScored: evaluation.scored,
				CatchAll:         isCatchAllRules(decision.Rules),
			})
		}
	}

	if len(results) == 0 {
		logging.Infof("No decision matched")
		return nil, nil
	}

	return e.selectBestDecision(results), nil
}

func (e *DecisionEngine) evaluateDecisionWithSignals(
	decision *config.Decision,
	signals *SignalMatches,
	legacy bool,
) nodeEvaluation {
	if decision.Rules.IsEmpty() {
		return nodeEvaluation{state: evaluationTrue, scored: true}
	}
	evaluation, _ := e.evalNode(decision.Rules, signals, legacy, false)
	return evaluation
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
	legacy bool,
	withTrace bool,
) (nodeEvaluation, *TraceNode) {
	if node.IsLeaf() {
		evaluation := e.evalLeaf(node, signals, legacy)
		if !withTrace {
			return evaluation, nil
		}
		return evaluation, &TraceNode{
			NodeType:         "leaf",
			SignalType:       node.Type,
			SignalName:       node.Name,
			Label:            node.Label,
			Matched:          evaluation.state == evaluationTrue,
			Confidence:       evaluation.confidence,
			ConfidenceScored: evaluation.scored,
		}
	}

	switch strings.ToUpper(node.Operator) {
	case "AND":
		return e.evalAND(node.Conditions, signals, legacy, withTrace)
	case "NOT":
		return e.evalNOT(node.Conditions, signals, legacy, withTrace)
	default: // OR
		return e.evalOR(node.Conditions, signals, legacy, withTrace)
	}
}

func newTraceNode(nodeType string, withTrace bool) *TraceNode {
	if !withTrace {
		return nil
	}
	return &TraceNode{NodeType: nodeType}
}

func (t *TraceNode) addChild(child *TraceNode) {
	if t != nil {
		t.Children = append(t.Children, child)
	}
}

func (t *TraceNode) finish(evaluation nodeEvaluation) {
	if t == nil {
		return
	}
	t.Matched = evaluation.state == evaluationTrue
	t.Confidence = evaluation.confidence
	t.ConfidenceScored = evaluation.scored
}

// evalLeaf evaluates a single signal condition (leaf node).
func (e *DecisionEngine) evalLeaf(
	node config.RuleNode,
	signals *SignalMatches,
	legacy bool,
) nodeEvaluation {
	normalizedType := strings.ToLower(strings.TrimSpace(node.Type))
	matched, supported := e.matchesSignalType(normalizedType, node.Name, signals)
	if !supported {
		return nodeEvaluation{state: evaluationFalse}
	}
	if node.Predicate != nil {
		return evaluatePredicateLeaf(node, normalizedType, signals, legacy)
	}
	if signalFailed(signals, normalizedType, node.Name) && !legacy &&
		(!matched || signalErrorMatch(signals, normalizedType, node.Name)) {
		return nodeEvaluation{state: evaluationUnknown}
	}
	if !matched {
		return nodeEvaluation{state: evaluationFalse}
	}
	confidence, scored := signalConfidence(signals.SignalConfidences, normalizedType, node.Name)
	return nodeEvaluation{
		state:        evaluationTrue,
		confidence:   confidence,
		scored:       scored,
		matchedRules: []string{formatMatchedRule(node)},
	}
}

func evaluatePredicateLeaf(
	node config.RuleNode,
	normalizedType string,
	signals *SignalMatches,
	legacy bool,
) nodeEvaluation {
	value, available := signalPredicateValue(signals, normalizedType, node.Name, node.Label)
	if available {
		if numericPredicateMatches(value, node.Predicate) {
			return nodeEvaluation{state: evaluationTrue, confidence: 1, matchedRules: []string{formatMatchedRule(node)}}
		}
		return nodeEvaluation{state: evaluationFalse}
	}
	if !signalFailed(signals, normalizedType, node.Name) {
		return nodeEvaluation{state: evaluationFalse}
	}
	if !legacy {
		return nodeEvaluation{state: evaluationUnknown}
	}
	if strings.EqualFold(strings.TrimSpace(node.OnError), "match") {
		return nodeEvaluation{state: evaluationTrue, confidence: 1, matchedRules: []string{formatMatchedRule(node)}}
	}
	return nodeEvaluation{state: evaluationFalse}
}

func signalFailed(signals *SignalMatches, signalType, name string) bool {
	if signals == nil || signals.SignalErrors == nil {
		return false
	}
	_, failed := signals.SignalErrors[fmt.Sprintf("%s:%s", signalType, name)]
	return failed
}

func signalErrorMatch(signals *SignalMatches, signalType, name string) bool {
	if signals == nil || signals.SignalErrorMatches == nil {
		return false
	}
	return signals.SignalErrorMatches[fmt.Sprintf("%s:%s", signalType, name)]
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
	legacy bool,
	withTrace bool,
) (nodeEvaluation, *TraceNode) {
	trace := newTraceNode("AND", withTrace)
	evaluation := nodeEvaluation{state: evaluationTrue, scored: true}
	if len(children) == 0 {
		trace.finish(evaluation)
		return evaluation, trace
	}
	totalConfidence := 0.0
	matchedCount := 0
	for _, child := range children {
		childEvaluation, childTrace := e.evalNode(child, signals, legacy, withTrace)
		trace.addChild(childTrace)
		if childEvaluation.state == evaluationFalse {
			evaluation = nodeEvaluation{state: evaluationFalse}
			trace.finish(evaluation)
			return evaluation, trace
		}
		if childEvaluation.state == evaluationUnknown {
			evaluation.state = evaluationUnknown
			continue
		}
		totalConfidence += childEvaluation.confidence
		matchedCount++
		evaluation.scored = evaluation.scored && childEvaluation.scored
		evaluation.matchedRules = append(evaluation.matchedRules, childEvaluation.matchedRules...)
	}
	if evaluation.state == evaluationUnknown {
		evaluation.scored = false
	} else if matchedCount > 0 {
		evaluation.confidence = totalConfidence / float64(matchedCount)
	}
	trace.finish(evaluation)
	return evaluation, trace
}

// evalOR returns true when at least one child matches; returns the best-confidence match.
func (e *DecisionEngine) evalOR(
	children []config.RuleNode,
	signals *SignalMatches,
	legacy bool,
	withTrace bool,
) (nodeEvaluation, *TraceNode) {
	trace := newTraceNode("OR", withTrace)
	evaluation := nodeEvaluation{state: evaluationFalse}
	unknown := false
	for _, child := range children {
		childEvaluation, childTrace := e.evalNode(child, signals, legacy, withTrace)
		trace.addChild(childTrace)
		switch childEvaluation.state {
		case evaluationUnknown:
			unknown = true
		case evaluationTrue:
			if evaluation.state != evaluationTrue || childEvaluation.confidence > evaluation.confidence {
				evaluation = childEvaluation
			}
		}
	}
	if evaluation.state != evaluationTrue && unknown {
		evaluation.state = evaluationUnknown
	}
	trace.finish(evaluation)
	return evaluation, trace
}

// evalNOT is a strictly unary operator: it negates the result of its single child.
// Configuration errors (0 or 2+ children) are treated as non-matching.
func (e *DecisionEngine) evalNOT(
	children []config.RuleNode,
	signals *SignalMatches,
	legacy bool,
	withTrace bool,
) (nodeEvaluation, *TraceNode) {
	trace := newTraceNode("NOT", withTrace)
	if len(children) != 1 {
		logging.Warnf("NOT operator requires exactly 1 child, got %d — treating as non-match", len(children))
		return nodeEvaluation{state: evaluationFalse}, trace
	}
	childEvaluation, childTrace := e.evalNode(children[0], signals, legacy, withTrace)
	trace.addChild(childTrace)
	var evaluation nodeEvaluation
	switch childEvaluation.state {
	case evaluationFalse:
		evaluation = nodeEvaluation{state: evaluationTrue, confidence: 1, matchedRules: childEvaluation.matchedRules}
	case evaluationUnknown:
		evaluation = nodeEvaluation{state: evaluationUnknown}
	default:
		evaluation = nodeEvaluation{
			state:        evaluationFalse,
			confidence:   childEvaluation.confidence,
			scored:       childEvaluation.scored,
			matchedRules: childEvaluation.matchedRules,
		}
	}
	trace.finish(evaluation)
	return evaluation, trace
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
