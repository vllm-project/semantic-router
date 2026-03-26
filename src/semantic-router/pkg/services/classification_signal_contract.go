package services

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

var matchedSignalResolvers = map[string]func(*MatchedSignals) *[]string{
	config.SignalTypeKeyword:      func(target *MatchedSignals) *[]string { return &target.Keywords },
	config.SignalTypeEmbedding:    func(target *MatchedSignals) *[]string { return &target.Embeddings },
	config.SignalTypeDomain:       func(target *MatchedSignals) *[]string { return &target.Domains },
	config.SignalTypeFactCheck:    func(target *MatchedSignals) *[]string { return &target.FactCheck },
	config.SignalTypeUserFeedback: func(target *MatchedSignals) *[]string { return &target.UserFeedback },
	config.SignalTypePreference:   func(target *MatchedSignals) *[]string { return &target.Preferences },
	config.SignalTypeLanguage:     func(target *MatchedSignals) *[]string { return &target.Language },
	config.SignalTypeContext:      func(target *MatchedSignals) *[]string { return &target.Context },
	config.SignalTypeStructure:    func(target *MatchedSignals) *[]string { return &target.Structure },
	config.SignalTypeComplexity:   func(target *MatchedSignals) *[]string { return &target.Complexity },
	config.SignalTypeModality:     func(target *MatchedSignals) *[]string { return &target.Modality },
	config.SignalTypeAuthz:        func(target *MatchedSignals) *[]string { return &target.Authz },
	config.SignalTypeJailbreak:    func(target *MatchedSignals) *[]string { return &target.Jailbreak },
	config.SignalTypePII:          func(target *MatchedSignals) *[]string { return &target.PII },
	config.SignalTypeKB:           func(target *MatchedSignals) *[]string { return &target.KB },
	config.SignalTypeProjection:   func(target *MatchedSignals) *[]string { return &target.Projection },
}

// IntentRequest represents a request for intent classification
type IntentRequest struct {
	Text    string         `json:"text"`
	Options *IntentOptions `json:"options,omitempty"`
}

// IntentOptions contains options for intent classification
type IntentOptions struct {
	ReturnProbabilities bool    `json:"return_probabilities,omitempty"`
	ConfidenceThreshold float64 `json:"confidence_threshold,omitempty"`
	IncludeExplanation  bool    `json:"include_explanation,omitempty"`
	EvaluateAllSignals  bool    `json:"evaluate_all_signals,omitempty"` // Force evaluate all configured signals (for eval scenarios)
}

// MatchedSignals represents all matched signals from signal evaluation
type MatchedSignals struct {
	Keywords     []string `json:"keywords,omitempty"`
	Embeddings   []string `json:"embeddings,omitempty"`
	Domains      []string `json:"domains,omitempty"`
	FactCheck    []string `json:"fact_check,omitempty"`
	UserFeedback []string `json:"user_feedback,omitempty"`
	Preferences  []string `json:"preferences,omitempty"`
	Language     []string `json:"language,omitempty"`
	Context      []string `json:"context,omitempty"`
	Structure    []string `json:"structure,omitempty"`
	Complexity   []string `json:"complexity,omitempty"`
	Modality     []string `json:"modality,omitempty"`
	Authz        []string `json:"authz,omitempty"`
	Jailbreak    []string `json:"jailbreak,omitempty"`
	PII          []string `json:"pii,omitempty"`
	KB           []string `json:"kb,omitempty"`
	Projection   []string `json:"projection,omitempty"`
}

// DecisionResult represents the result of decision evaluation
type DecisionResult struct {
	DecisionName string   `json:"decision_name"`
	Confidence   float64  `json:"confidence"`
	MatchedRules []string `json:"matched_rules"`
}

// EvalDecisionResult represents the decision result for eval scenarios (without confidence)
type EvalDecisionResult struct {
	DecisionName     string          `json:"decision_name"`
	UsedSignals      *MatchedSignals `json:"used_signals"`      // Signals used by this decision (from decision rules)
	MatchedSignals   *MatchedSignals `json:"matched_signals"`   // Signals that matched
	UnmatchedSignals *MatchedSignals `json:"unmatched_signals"` // Signals that didn't match
}

// EvalResponse represents the eval classification response with comprehensive signal information.
type EvalResponse struct {
	OriginalText      string                                  `json:"original_text"` // The original query text
	DecisionResult    *EvalDecisionResult                     `json:"decision_result,omitempty"`
	RecommendedModels []string                                `json:"recommended_models,omitempty"` // All models from matched decision's modelRefs
	RoutingDecision   string                                  `json:"routing_decision,omitempty"`
	Metrics           *classification.SignalMetricsCollection `json:"metrics"`                      // Performance and confidence for each signal
	SignalConfidences map[string]float64                      `json:"signal_confidences,omitempty"` // Real ML confidence scores per signal, e.g. "domain:economics" → 0.81
	SignalValues      map[string]float64                      `json:"signal_values,omitempty"`      // Raw signal values per signal when exposed, e.g. "structure:many_questions" → 4
}

// IntentResponse represents the response from intent classification
type IntentResponse struct {
	Classification   Classification     `json:"classification"`
	Probabilities    map[string]float64 `json:"probabilities,omitempty"`
	RecommendedModel string             `json:"recommended_model,omitempty"`
	RoutingDecision  string             `json:"routing_decision,omitempty"`

	// Signal-driven fields
	MatchedSignals *MatchedSignals `json:"matched_signals,omitempty"`
	DecisionResult *DecisionResult `json:"decision_result,omitempty"`
}

// Classification represents basic classification result
type Classification struct {
	Category         string  `json:"category"`
	Confidence       float64 `json:"confidence"`
	ProcessingTimeMs int64   `json:"processing_time_ms"`
}

// buildIntentResponseFromSignals builds an IntentResponse from signals and decision result
func (s *ClassificationService) buildIntentResponseFromSignals(
	signals *classification.SignalResults,
	decisionResult *decision.DecisionResult,
	category string,
	confidence float64,
	processingTime int64,
	req IntentRequest,
) *IntentResponse {
	response := &IntentResponse{
		Classification: Classification{
			Category:         category,
			Confidence:       confidence,
			ProcessingTimeMs: processingTime,
		},
	}

	populateIntentProbabilities(response, category, confidence, req.Options)
	response.RecommendedModel = s.resolveRecommendedModel(decisionResult, category, confidence)
	response.RoutingDecision = s.resolveRoutingDecision(decisionResult, confidence, req.Options)
	if signals != nil {
		response.MatchedSignals = buildMatchedSignals(signals)
	}
	if decisionPayload := buildDecisionResultPayload(decisionResult); decisionPayload != nil {
		response.DecisionResult = decisionPayload
	}

	return response
}

// ClassifyIntentForEval performs intent classification specifically for evaluation scenarios
// This method forces evaluation of all signals and returns comprehensive signal information
func (s *ClassificationService) ClassifyIntentForEval(req IntentRequest) (*EvalResponse, error) {
	if req.Text == "" {
		return nil, fmt.Errorf("text cannot be empty")
	}

	if s.classifier == nil {
		return &EvalResponse{
			OriginalText: req.Text,
			Metrics:      &classification.SignalMetricsCollection{},
		}, nil
	}

	signals := s.classifier.EvaluateAllSignalsWithForceOption(req.Text, true)

	var decisionResult *decision.DecisionResult
	var err error
	if s.config != nil && len(s.config.Decisions) > 0 {
		decisionResult, err = s.classifier.EvaluateDecisionWithEngine(signals)
		if err != nil && !strings.Contains(err.Error(), "no decisions configured") {
			logging.Warnf("Decision evaluation failed: %v", err)
		}
	}

	return s.buildEvalResponse(req.Text, signals, decisionResult), nil
}

func buildMatchedSignals(signals *classification.SignalResults) *MatchedSignals {
	if signals == nil {
		return &MatchedSignals{}
	}

	return &MatchedSignals{
		Keywords:     signals.MatchedKeywordRules,
		Embeddings:   signals.MatchedEmbeddingRules,
		Domains:      signals.MatchedDomainRules,
		FactCheck:    signals.MatchedFactCheckRules,
		UserFeedback: signals.MatchedUserFeedbackRules,
		Preferences:  signals.MatchedPreferenceRules,
		Language:     signals.MatchedLanguageRules,
		Context:      signals.MatchedContextRules,
		Structure:    signals.MatchedStructureRules,
		Complexity:   signals.MatchedComplexityRules,
		Modality:     signals.MatchedModalityRules,
		Authz:        signals.MatchedAuthzRules,
		Jailbreak:    signals.MatchedJailbreakRules,
		PII:          signals.MatchedPIIRules,
		KB:           signals.MatchedKBRules,
		Projection:   signals.MatchedProjectionRules,
	}
}

// buildEvalResponse builds an EvalResponse from signal results and decision result
func (s *ClassificationService) buildEvalResponse(
	text string,
	signals *classification.SignalResults,
	decisionResult *decision.DecisionResult,
) *EvalResponse {
	response := &EvalResponse{
		OriginalText:      text,
		Metrics:           signals.Metrics,
		SignalConfidences: signals.SignalConfidences,
		SignalValues:      signals.SignalValues,
	}

	matchedSignals := buildMatchedSignals(signals)
	unmatchedSignals := s.getUnmatchedSignals(signals)

	if decisionResult != nil && decisionResult.Decision != nil {
		usedSignals := s.extractUsedSignalsFromDecision(decisionResult.Decision)

		response.DecisionResult = &EvalDecisionResult{
			DecisionName:     decisionResult.Decision.Name,
			UsedSignals:      usedSignals,
			MatchedSignals:   matchedSignals,
			UnmatchedSignals: unmatchedSignals,
		}

		if len(decisionResult.Decision.ModelRefs) > 0 {
			models := make([]string, 0, len(decisionResult.Decision.ModelRefs))
			for _, modelRef := range decisionResult.Decision.ModelRefs {
				models = append(models, modelRef.Model)
			}
			response.RecommendedModels = models
			response.RoutingDecision = decisionResult.Decision.Name
		}
	} else {
		response.DecisionResult = &EvalDecisionResult{
			DecisionName:     "",
			UsedSignals:      &MatchedSignals{},
			MatchedSignals:   matchedSignals,
			UnmatchedSignals: unmatchedSignals,
		}
	}

	return response
}

// extractUsedSignalsFromDecision extracts all signals used in a decision's rule configuration
// This includes ALL signals defined in the decision rules, not just the ones that matched
func (s *ClassificationService) extractUsedSignalsFromDecision(decision *config.Decision) *MatchedSignals {
	usedSignals := &MatchedSignals{}
	s.extractSignalsFromRuleCombination(decision.Rules, usedSignals)
	return usedSignals
}

// extractSignalsFromRuleCombination recursively extracts signals from a rule combination
func (s *ClassificationService) extractSignalsFromRuleCombination(rules config.RuleCombination, usedSignals *MatchedSignals) {
	if rules.IsLeaf() {
		appendSignalToMatchedSignals(usedSignals, rules.Type, rules.Name)
	}
	for _, condition := range rules.Conditions {
		s.extractSignalsFromRuleCombination(condition, usedSignals)
	}
}

func appendSignalName(target *[]string, signalName string) {
	if signalName == "" {
		return
	}
	if !contains(*target, signalName) {
		*target = append(*target, signalName)
	}
}

// contains checks if a string slice contains a specific string
func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

// getUnmatchedSignals returns all configured signals that were not matched
func (s *ClassificationService) getUnmatchedSignals(signals *classification.SignalResults) *MatchedSignals {
	unmatched := &MatchedSignals{}

	if s.classifier == nil || s.config == nil {
		return unmatched
	}
	cfg := s.classifier.Config
	collectUnmatchedRuleNames(&unmatched.Keywords, cfg.KeywordRules, signals.MatchedKeywordRules, func(rule config.KeywordRule) string { return rule.Name })
	collectUnmatchedRuleNames(&unmatched.Embeddings, cfg.EmbeddingRules, signals.MatchedEmbeddingRules, func(rule config.EmbeddingRule) string { return rule.Name })
	collectUnmatchedRuleNames(&unmatched.Domains, cfg.Categories, signals.MatchedDomainRules, func(category config.Category) string { return category.Name })
	collectUnmatchedRuleNames(&unmatched.FactCheck, cfg.FactCheckRules, signals.MatchedFactCheckRules, func(rule config.FactCheckRule) string { return rule.Name })
	collectUnmatchedRuleNames(&unmatched.UserFeedback, cfg.UserFeedbackRules, signals.MatchedUserFeedbackRules, func(rule config.UserFeedbackRule) string { return rule.Name })
	collectUnmatchedRuleNames(&unmatched.Preferences, cfg.PreferenceRules, signals.MatchedPreferenceRules, func(rule config.PreferenceRule) string { return rule.Name })
	collectUnmatchedRuleNames(&unmatched.Language, cfg.LanguageRules, signals.MatchedLanguageRules, func(rule config.LanguageRule) string { return rule.Name })
	collectUnmatchedRuleNames(&unmatched.Context, cfg.ContextRules, signals.MatchedContextRules, func(rule config.ContextRule) string { return rule.Name })
	collectUnmatchedRuleNames(&unmatched.Structure, cfg.StructureRules, signals.MatchedStructureRules, func(rule config.StructureRule) string { return rule.Name })
	collectUnmatchedRuleNames(&unmatched.Complexity, cfg.ComplexityRules, signals.MatchedComplexityRules, func(rule config.ComplexityRule) string { return rule.Name })
	collectUnmatchedRuleNames(&unmatched.Modality, cfg.ModalityRules, signals.MatchedModalityRules, func(rule config.ModalityRule) string { return rule.Name })
	collectUnmatchedAuthzRules(&unmatched.Authz, cfg.GetRoleBindings(), signals.MatchedAuthzRules)
	collectUnmatchedRuleNames(&unmatched.Jailbreak, cfg.JailbreakRules, signals.MatchedJailbreakRules, func(rule config.JailbreakRule) string { return rule.Name })
	collectUnmatchedRuleNames(&unmatched.PII, cfg.PIIRules, signals.MatchedPIIRules, func(rule config.PIIRule) string { return rule.Name })
	collectUnmatchedProjectionOutputs(&unmatched.Projection, cfg.Projections.Mappings, signals.MatchedProjectionRules)

	return unmatched
}

func populateIntentProbabilities(
	response *IntentResponse,
	category string,
	confidence float64,
	options *IntentOptions,
) {
	if options == nil || !options.ReturnProbabilities {
		return
	}
	response.Probabilities = map[string]float64{category: confidence}
}

func (s *ClassificationService) resolveRecommendedModel(
	decisionResult *decision.DecisionResult,
	category string,
	confidence float64,
) string {
	if decisionResult != nil && decisionResult.Decision != nil && len(decisionResult.Decision.ModelRefs) > 0 {
		modelRef := decisionResult.Decision.ModelRefs[0]
		if modelRef.LoRAName != "" {
			return modelRef.LoRAName
		}
		return modelRef.Model
	}
	return s.getRecommendedModel(category, confidence)
}

func (s *ClassificationService) resolveRoutingDecision(
	decisionResult *decision.DecisionResult,
	confidence float64,
	options *IntentOptions,
) string {
	if decisionResult != nil && decisionResult.Decision != nil {
		return decisionResult.Decision.Name
	}
	return s.getRoutingDecision(confidence, options)
}

func buildDecisionResultPayload(decisionResult *decision.DecisionResult) *DecisionResult {
	if decisionResult == nil || decisionResult.Decision == nil {
		return nil
	}
	return &DecisionResult{
		DecisionName: decisionResult.Decision.Name,
		Confidence:   decisionResult.Confidence,
		MatchedRules: decisionResult.MatchedRules,
	}
}

func appendSignalToMatchedSignals(target *MatchedSignals, signalType string, signalName string) {
	resolver, ok := matchedSignalResolvers[strings.ToLower(strings.TrimSpace(signalType))]
	if !ok {
		return
	}
	appendSignalName(resolver(target), strings.TrimSpace(signalName))
}

func collectUnmatchedRuleNames[T any](target *[]string, rules []T, matched []string, nameFn func(T) string) {
	matchedSet := makeStringSet(matched)
	for _, rule := range rules {
		name := strings.TrimSpace(nameFn(rule))
		if name == "" || matchedSet[name] {
			continue
		}
		*target = append(*target, name)
	}
}

func collectUnmatchedAuthzRules(target *[]string, rules []config.RoleBinding, matched []string) {
	matchedSet := makeStringSet(matched)
	for _, rule := range rules {
		if matchedSet[rule.Role] || matchedSet[rule.Name] {
			continue
		}
		name := strings.TrimSpace(rule.Name)
		if name == "" {
			name = strings.TrimSpace(rule.Role)
		}
		if name != "" {
			*target = append(*target, name)
		}
	}
}

func collectUnmatchedProjectionOutputs(
	target *[]string,
	mappings []config.ProjectionMapping,
	matched []string,
) {
	matchedSet := makeStringSet(matched)
	for _, mapping := range mappings {
		for _, output := range mapping.Outputs {
			if output.Name == "" || matchedSet[output.Name] {
				continue
			}
			*target = append(*target, output.Name)
		}
	}
}

func makeStringSet(values []string) map[string]bool {
	set := make(map[string]bool, len(values))
	for _, value := range values {
		set[value] = true
	}
	return set
}
