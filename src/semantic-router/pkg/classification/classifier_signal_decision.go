package classification

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// EvaluateDecisionWithEngine evaluates all decisions using pre-computed signals.
// Accepts SignalResults to avoid duplicate signal computation.
func (c *Classifier) EvaluateDecisionWithEngine(signals *SignalResults) (*decision.DecisionResult, error) {
	result, _, err := c.evaluateDecisionInternal(signals, false, nil)
	return result, err
}

// EvaluateDecisionWithEngineForDecisions evaluates only the supplied decision candidates.
func (c *Classifier) EvaluateDecisionWithEngineForDecisions(signals *SignalResults, decisions []config.Decision) (*decision.DecisionResult, error) {
	result, _, err := c.evaluateDecisionInternal(signals, false, decisions)
	return result, err
}

// EvaluateDecisionWithEngineAndTrace evaluates decisions and returns a
// per-decision trace tree that mirrors the boolean expression structure.
func (c *Classifier) EvaluateDecisionWithEngineAndTrace(signals *SignalResults) (*decision.DecisionResult, []decision.DecisionTrace, error) {
	return c.evaluateDecisionInternal(signals, true, nil)
}

func (c *Classifier) evaluateDecisionInternal(signals *SignalResults, trace bool, candidates []config.Decision) (*decision.DecisionResult, []decision.DecisionTrace, error) {
	decisions := c.Config.Decisions
	if candidates != nil {
		decisions = candidates
	}
	if len(decisions) == 0 {
		return nil, nil, fmt.Errorf("no decisions configured")
	}

	logging.Debugf("Signal evaluation results: keyword=%v, embedding=%v, domain=%v, fact_check=%v, user_feedback=%v, reask=%v, preference=%v, language=%v, context=%v, structure=%v, complexity=%v, modality=%v, authz=%v, jailbreak=%v, pii=%v, kb=%v, conversation=%v, event=%v",
		signals.MatchedKeywordRules, signals.MatchedEmbeddingRules, signals.MatchedDomainRules,
		signals.MatchedFactCheckRules, signals.MatchedUserFeedbackRules, signals.MatchedReaskRules, signals.MatchedPreferenceRules,
		signals.MatchedLanguageRules, signals.MatchedContextRules, signals.MatchedStructureRules,
		signals.MatchedComplexityRules, signals.MatchedModalityRules, signals.MatchedAuthzRules,
		signals.MatchedJailbreakRules, signals.MatchedPIIRules, signals.MatchedKBRules,
		signals.MatchedConversationRules, signals.MatchedEventRules)

	engine := decision.NewDecisionEngine(
		c.Config.KeywordRules,
		c.Config.EmbeddingRules,
		c.Config.Categories,
		decisions,
		c.Config.Strategy,
	)

	sm := &decision.SignalMatches{
		KeywordRules:      signals.MatchedKeywordRules,
		EmbeddingRules:    signals.MatchedEmbeddingRules,
		DomainRules:       signals.MatchedDomainRules,
		FactCheckRules:    signals.MatchedFactCheckRules,
		UserFeedbackRules: signals.MatchedUserFeedbackRules,
		ReaskRules:        signals.MatchedReaskRules,
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
		ConversationRules: signals.MatchedConversationRules,
		EventRules:        signals.MatchedEventRules,
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
