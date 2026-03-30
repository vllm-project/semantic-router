package classification

import (
	"fmt"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// getUsedSignals analyzes all decisions and returns which signals (type:name) are actually used
// This allows us to skip evaluation of unused signals for performance optimization
// Returns a map with keys in format "type:name" (e.g., "keyword:math_keywords")
func (c *Classifier) getUsedSignals() map[string]bool {
	usedSignals := make(map[string]bool)

	// Analyze all decisions to find which signals are referenced
	for _, decision := range c.Config.Decisions {
		c.analyzeRuleCombination(decision.Rules, usedSignals)
	}
	c.expandProjectionDependencies(usedSignals)

	return usedSignals
}

// collectSignalKeys adds signal keys for a slice of items using a name-extraction function.
func collectSignalKeys[T any](signals map[string]bool, signalType string, items []T, getName func(T) string) {
	for _, item := range items {
		signals[strings.ToLower(signalType+":"+getName(item))] = true
	}
}

// getAllSignalTypes returns a map containing all configured signal types
// This is used when forceEvaluateAll is true to evaluate all signals regardless of decision usage
func (c *Classifier) getAllSignalTypes() map[string]bool {
	allSignals := make(map[string]bool)

	collectSignalKeys(allSignals, config.SignalTypeKeyword, c.Config.KeywordRules, func(r config.KeywordRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypeEmbedding, c.Config.EmbeddingRules, func(r config.EmbeddingRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypeDomain, c.Config.Categories, func(r config.Category) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypeFactCheck, c.Config.FactCheckRules, func(r config.FactCheckRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypeUserFeedback, c.Config.UserFeedbackRules, func(r config.UserFeedbackRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypePreference, c.Config.PreferenceRules, func(r config.PreferenceRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypeLanguage, c.Config.LanguageRules, func(r config.LanguageRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypeContext, c.Config.ContextRules, func(r config.ContextRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypeStructure, c.Config.StructureRules, func(r config.StructureRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypeComplexity, c.Config.ComplexityRules, func(r config.ComplexityRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypeModality, c.Config.ModalityRules, func(r config.ModalityRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypeAuthz, c.Config.GetRoleBindings(), func(rb config.RoleBinding) string { return rb.Role })
	collectSignalKeys(allSignals, config.SignalTypeJailbreak, c.Config.JailbreakRules, func(r config.JailbreakRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypePII, c.Config.PIIRules, func(r config.PIIRule) string { return r.Name })
	collectSignalKeys(allSignals, config.SignalTypeKB, c.Config.KBRules, func(r config.KBSignalRule) string { return r.Name })
	for _, mapping := range c.Config.Projections.Mappings {
		for _, output := range mapping.Outputs {
			allSignals[strings.ToLower(config.SignalTypeProjection+":"+output.Name)] = true
		}
	}
	c.expandProjectionDependencies(allSignals)

	return allSignals
}

// SignalMetrics contains performance and probability metrics for a single signal
type SignalMetrics struct {
	ExecutionTimeMs float64 `json:"execution_time_ms"` // Execution time in milliseconds
	Confidence      float64 `json:"confidence"`        // Confidence score (0.0-1.0), 0 if not applicable
}

// SignalResults contains all evaluated signal results
type SignalResults struct {
	MatchedKeywordRules      []string
	MatchedKeywords          []string // The actual keywords that matched (not rule names)
	MatchedEmbeddingRules    []string
	MatchedDomainRules       []string
	MatchedFactCheckRules    []string // "needs_fact_check" or "no_fact_check_needed"
	MatchedUserFeedbackRules []string // "satisfied", "need_clarification", "wrong_answer", "want_different"
	MatchedPreferenceRules   []string // Route preference names matched via external LLM
	MatchedLanguageRules     []string // Language codes: "en", "es", "zh", "fr", etc.
	MatchedContextRules      []string // Matched context rule names (e.g. "low_token_count")
	TokenCount               int      // Total token count
	MatchedStructureRules    []string // Matched structure rule names (e.g. "many_questions")
	MatchedComplexityRules   []string // Matched complexity rules with difficulty level (e.g. "code_complexity:hard")
	MatchedModalityRules     []string // Matched modality: "AR", "DIFFUSION", or "BOTH"
	MatchedAuthzRules        []string // Matched authz role names for user-level RBAC routing
	MatchedJailbreakRules    []string // Matched jailbreak rule names (confidence >= threshold)
	MatchedPIIRules          []string // Matched PII rule names (denied PII types detected)
	MatchedKBRules           []string
	KBClassifierResults      map[string]*KBClassifyResult
	KBMetricValues           map[string]float64
	MatchedProjectionRules   []string // Matched derived routing outputs from routing.projections.mappings
	ProjectionScores         map[string]float64

	// Jailbreak detection metadata (populated when jailbreak signal is evaluated)
	JailbreakDetected   bool    // Whether any jailbreak was detected (across all rules)
	JailbreakType       string  // Type of the detected jailbreak (from highest-confidence detection)
	JailbreakConfidence float32 // Confidence of the detected jailbreak

	// PII detection metadata (populated when PII signal is evaluated)
	PIIDetected bool     // Whether any PII was detected
	PIIEntities []string // Detected PII entity types (e.g., "EMAIL_ADDRESS", "PERSON")

	SignalConfidences map[string]float64 // Real confidence scores per signal, e.g. "embedding:ai" → 0.88
	SignalValues      map[string]float64 // Raw signal values per signal when the evaluator exposes them, e.g. "structure:many_questions" → 4

	// Signal metrics (only populated in eval mode)
	Metrics *SignalMetricsCollection
}

// SignalMetricsCollection contains metrics for all signal types
type SignalMetricsCollection struct {
	Keyword      SignalMetrics `json:"keyword"`
	Embedding    SignalMetrics `json:"embedding"`
	Domain       SignalMetrics `json:"domain"`
	FactCheck    SignalMetrics `json:"fact_check"`
	UserFeedback SignalMetrics `json:"user_feedback"`
	Preference   SignalMetrics `json:"preference"`
	Language     SignalMetrics `json:"language"`
	Context      SignalMetrics `json:"context"`
	Structure    SignalMetrics `json:"structure"`
	Complexity   SignalMetrics `json:"complexity"`
	Modality     SignalMetrics `json:"modality"`
	Authz        SignalMetrics `json:"authz"`
	Jailbreak    SignalMetrics `json:"jailbreak"`
	PII          SignalMetrics `json:"pii"`
	KB           SignalMetrics `json:"kb"`
}

// analyzeRuleCombination recursively traverses a rule tree to collect all referenced signals.
func (c *Classifier) analyzeRuleCombination(node config.RuleNode, usedSignals map[string]bool) {
	if node.IsLeaf() {
		t := strings.ToLower(strings.TrimSpace(node.Type))
		n := strings.ToLower(strings.TrimSpace(node.Name))
		usedSignals[t+":"+n] = true
		return
	}
	for _, child := range node.Conditions {
		c.analyzeRuleCombination(child, usedSignals)
	}
}

func (c *Classifier) expandProjectionDependencies(usedSignals map[string]bool) {
	if len(c.Config.Projections.Scores) == 0 || len(c.Config.Projections.Mappings) == 0 {
		return
	}

	scoreByName := make(map[string]config.ProjectionScore, len(c.Config.Projections.Scores))
	for _, score := range c.Config.Projections.Scores {
		scoreByName[score.Name] = score
	}

	sourceByOutput := make(map[string]string)
	for _, mapping := range c.Config.Projections.Mappings {
		for _, output := range mapping.Outputs {
			sourceByOutput[strings.ToLower(output.Name)] = mapping.Source
		}
	}

	for key := range usedSignals {
		if !strings.HasPrefix(key, config.SignalTypeProjection+":") {
			continue
		}
		outputName := strings.TrimPrefix(key, config.SignalTypeProjection+":")
		scoreName, ok := sourceByOutput[outputName]
		if !ok {
			continue
		}
		score, ok := scoreByName[scoreName]
		if !ok {
			continue
		}
		for _, input := range score.Inputs {
			if strings.EqualFold(input.Type, config.ProjectionInputKBMetric) {
				usedSignals[strings.ToLower(config.SignalTypeKB+":"+input.KB)] = true
				continue
			}
			usedSignals[strings.ToLower(input.Type+":"+input.Name)] = true
		}
	}
}

// isSignalTypeUsed checks if any signal of the given type is used in decisions
func isSignalTypeUsed(usedSignals map[string]bool, signalType string) bool {
	// Normalize signal type for comparison (all signals are normalized to lowercase)
	normalizedType := strings.ToLower(strings.TrimSpace(signalType))
	prefix := normalizedType + ":"

	for key := range usedSignals {
		// All signal keys are normalized to lowercase, so use case-insensitive comparison
		if strings.HasPrefix(strings.ToLower(strings.TrimSpace(key)), prefix) {
			return true
		}
	}
	return false
}

// EvaluateAllSignals evaluates all signal types and returns SignalResults
// This is the new method that includes fact_check signals
func (c *Classifier) EvaluateAllSignals(text string) *SignalResults {
	return c.EvaluateAllSignalsWithContext(text, text, nil, false, "", nil)
}

// EvaluateAllSignalsWithHeaders evaluates all signal types including the authz signal.
// The authz signal reads user identity and groups from request headers (x-authz-user-id,
// x-authz-user-groups) and evaluates role_bindings. Other signals are evaluated via
// EvaluateAllSignalsWithContext as before.
//
// Returns an error if authz evaluation fails (e.g., missing user identity header when
// role_bindings are configured). Errors are NOT swallowed — the caller must handle them.
// This prevents silent bypass of authz policies.
//
// headers: request headers from ext_proc (includes Authorino-injected authz headers)
//
// Optional trailing arguments (positional after imageURL):
//   - uncompressedText (string): original text before prompt compression
//   - skipCompressionSignals (map[string]bool): signal types that must use uncompressedText
func (c *Classifier) EvaluateAllSignalsWithHeaders(text string, contextText string, nonUserMessages []string, headers map[string]string, forceEvaluateAll bool, imageURL string, extra ...interface{}) (*SignalResults, error) {
	var uncompressedText string
	var skipCompressionSignals map[string]bool
	if len(extra) >= 2 {
		if s, ok := extra[0].(string); ok {
			uncompressedText = s
		}
		if m, ok := extra[1].(map[string]bool); ok {
			skipCompressionSignals = m
		}
	}
	results := c.EvaluateAllSignalsWithContext(text, contextText, nonUserMessages, forceEvaluateAll, uncompressedText, skipCompressionSignals, imageURL)

	// Evaluate authz signal if role bindings are configured and the signal type is used
	usedSignals := c.getUsedSignals()
	if forceEvaluateAll {
		usedSignals = c.getAllSignalTypes()
	}

	if isSignalTypeUsed(usedSignals, config.SignalTypeAuthz) && c.authzClassifier != nil {
		start := time.Now()
		userID := headers[c.authzUserIDHeader]
		userGroups := ParseUserGroups(headers[c.authzUserGroupsHeader])

		authzResult, err := c.authzClassifier.Classify(userID, userGroups)
		authzResult, err = applyAuthzFailOpenOnClassifyError(c.authzFailOpen, userID, authzResult, err)
		elapsed := time.Since(start)
		latencySeconds := elapsed.Seconds()

		// Record metrics
		results.Metrics.Authz.ExecutionTimeMs = float64(elapsed.Microseconds()) / 1000.0
		results.Metrics.Authz.Confidence = 1.0 // Rule-based, always 1.0

		if err != nil {
			// Do NOT swallow authz errors — propagate to caller.
			// A missing user identity header when role_bindings are configured is a hard failure,
			// not a signal that "didn't fire." Silent bypass is not allowed.
			logging.Errorf("[Authz Signal] classification failed: %v", err)
			metrics.RecordSignalExtraction(config.SignalTypeAuthz, "error", latencySeconds)
			return nil, fmt.Errorf("authz signal evaluation failed: %w", err)
		}

		for _, ruleName := range authzResult.MatchedRules {
			metrics.RecordSignalExtraction(config.SignalTypeAuthz, ruleName, latencySeconds)
			metrics.RecordSignalMatch(config.SignalTypeAuthz, ruleName)
		}
		results.MatchedAuthzRules = authzResult.MatchedRules

		logging.Debugf("[Signal Computation] Authz signal evaluation completed in %v", elapsed)
	} else if !isSignalTypeUsed(usedSignals, config.SignalTypeAuthz) {
		logging.Debugf("[Signal Computation] Authz signal not used in any decision, skipping evaluation")
	}

	return results, nil
}

// EvaluateAllSignalsWithForceOption evaluates signals with option to force evaluate all
// forceEvaluateAll: if true, evaluates all configured signals regardless of decision usage
func (c *Classifier) EvaluateAllSignalsWithForceOption(text string, forceEvaluateAll bool) *SignalResults {
	return c.EvaluateAllSignalsWithContext(text, text, nil, forceEvaluateAll, "", nil)
}

// EvaluateDecisionWithEngine evaluates all decisions using pre-computed signals
// Accepts SignalResults to avoid duplicate signal computation
func (c *Classifier) EvaluateDecisionWithEngine(signals *SignalResults) (*decision.DecisionResult, error) {
	result, _, err := c.evaluateDecisionInternal(signals, false)
	return result, err
}

// EvaluateDecisionWithEngineAndTrace evaluates decisions and returns a
// per-decision trace tree that mirrors the boolean expression structure.
func (c *Classifier) EvaluateDecisionWithEngineAndTrace(signals *SignalResults) (*decision.DecisionResult, []decision.DecisionTrace, error) {
	return c.evaluateDecisionInternal(signals, true)
}

func (c *Classifier) evaluateDecisionInternal(signals *SignalResults, trace bool) (*decision.DecisionResult, []decision.DecisionTrace, error) {
	if len(c.Config.Decisions) == 0 {
		return nil, nil, fmt.Errorf("no decisions configured")
	}

	logging.Debugf("Signal evaluation results: keyword=%v, embedding=%v, domain=%v, fact_check=%v, user_feedback=%v, preference=%v, language=%v, context=%v, structure=%v, complexity=%v, modality=%v, authz=%v, jailbreak=%v, pii=%v, kb=%v",
		signals.MatchedKeywordRules, signals.MatchedEmbeddingRules, signals.MatchedDomainRules,
		signals.MatchedFactCheckRules, signals.MatchedUserFeedbackRules, signals.MatchedPreferenceRules,
		signals.MatchedLanguageRules, signals.MatchedContextRules, signals.MatchedStructureRules,
		signals.MatchedComplexityRules, signals.MatchedModalityRules, signals.MatchedAuthzRules,
		signals.MatchedJailbreakRules, signals.MatchedPIIRules, signals.MatchedKBRules)

	engine := decision.NewDecisionEngine(
		c.Config.KeywordRules,
		c.Config.EmbeddingRules,
		c.Config.Categories,
		c.Config.Decisions,
		c.Config.Strategy,
	)

	sm := &decision.SignalMatches{
		KeywordRules:      signals.MatchedKeywordRules,
		EmbeddingRules:    signals.MatchedEmbeddingRules,
		DomainRules:       signals.MatchedDomainRules,
		FactCheckRules:    signals.MatchedFactCheckRules,
		UserFeedbackRules: signals.MatchedUserFeedbackRules,
		PreferenceRules:   signals.MatchedPreferenceRules,
		LanguageRules:     signals.MatchedLanguageRules,
		ContextRules:      signals.MatchedContextRules,
		StructureRules:    signals.MatchedStructureRules,
		ComplexityRules:   signals.MatchedComplexityRules,
		ModalityRules:     signals.MatchedModalityRules,
		SignalConfidences: signals.SignalConfidences,
		AuthzRules:        signals.MatchedAuthzRules,
		JailbreakRules:    signals.MatchedJailbreakRules,
		PIIRules:          signals.MatchedPIIRules,
		KBRules:           signals.MatchedKBRules,
		ProjectionRules:   signals.MatchedProjectionRules,
	}

	var result *decision.DecisionResult
	var traces []decision.DecisionTrace

	if trace {
		result, traces = engine.EvaluateDecisionsWithTrace(sm)
	} else {
		var err error
		result, err = engine.EvaluateDecisionsWithSignals(sm)
		if err != nil {
			return nil, nil, fmt.Errorf("decision evaluation failed: %w", err)
		}
	}

	if result == nil {
		return nil, traces, nil
	}

	result.MatchedKeywords = signals.MatchedKeywords

	logging.Debugf("Decision evaluation result: decision=%s, confidence=%.3f, matched_rules=%v, matched_keywords=%v",
		result.Decision.Name, result.Confidence, result.MatchedRules, result.MatchedKeywords)

	return result, traces, nil
}
