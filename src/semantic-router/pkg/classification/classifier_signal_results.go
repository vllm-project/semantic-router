package classification

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/projectiontrace"
)

// SignalMetrics contains performance and probability metrics for a single signal.
type SignalMetrics struct {
	Method              string  `json:"method,omitempty"`
	ExecutionTimeMs     float64 `json:"execution_time_ms"` // Execution time in milliseconds
	Confidence          float64 `json:"confidence"`        // Confidence score (0.0-1.0), 0 if not applicable
	ConfidenceAvailable *bool   `json:"confidence_available,omitempty"`
}

// DomainClassificationResult preserves one domain evaluation before routing
// thresholds select matches. A failed evaluation has no available confidence.
type DomainClassificationResult struct {
	Category            string
	Confidence          float64
	ConfidenceAvailable bool
}

// SignalResults contains all evaluated signal results.
type SignalResults struct {
	MatchedKeywordRules []string
	MatchedKeywords     []string // The actual keywords that matched (not rule names)
	// The matched references of each systemone and application signal type:
	// a rule's name when the rule matches, and "<rule>:<label>" for each of
	// its options that does.
	MatchedDomainRules        []string
	MatchedFactCheckRules     []string
	MatchedUserFeedbackRules  []string
	MatchedLanguageRules      []string // Language codes: "en", "es", "zh", "fr", etc.
	MatchedContextRules       []string // Matched context rule names (e.g. "low_token_count")
	TokenCount                int      // Total token count
	MatchedComplexityRules    []string
	MatchedModalityRules      []string
	MatchedAuthzRules         []string // Matched authz role names for user-level RBAC routing
	MatchedJailbreakRules     []string
	MatchedSafetyRules        []string
	MatchedPIIRules           []string
	MatchedTurnRelationRules  []string
	MatchedHallucinationRules []string // Response stage only
	MatchedRefusalRules       []string // Response stage only
	MatchedConversationRules  []string
	MatchedMetadataRules      []string // Matched untrusted request metadata rules
	MatchedInputModalityRules []string // Matched structural input-modality presence rules
	MatchedSystemOneRules     []string
	MatchedProjectionRules    []string // Matched derived routing outputs from routing.projections.mappings
	ProjectionScores          map[string]float64
	ProjectionTrace           *projectiontrace.Trace // Explainability payload for projections (replay / dashboard)

	// Nil means no domains rule was asked for this request.
	DomainClassification *DomainClassificationResult

	// Jailbreak detection metadata (populated when jailbreak signal is evaluated)
	JailbreakDetected       bool    // Whether any jailbreak rule matched
	JailbreakType           string  // The label of the detection, or unscanned
	JailbreakConfidence     float32 // Highest observed probability of an attack, available even without a match
	JailbreakScoreAvailable bool

	// PII detection metadata (populated when PII signal is evaluated)
	// PIIContentVerified is true only after complete, successful classification with no personal entities.
	PIIContentVerified bool
	PIIEvidence        []PrivacyEvidence
	PIIDetected        bool     // Whether any PII was detected
	PIIEntities        []string // Detected PII entity types (e.g., "EMAIL_ADDRESS", "PERSON")

	SignalConfidences  map[string]float64             // Real confidence scores per signal, e.g. "embedding:ai" → 0.88
	SignalValues       map[string]float64             // Raw signal values per signal when the evaluator exposes them, e.g. "keyword:many_questions" → 4
	SignalErrors       map[string]string              // Signal evaluation errors keyed by "type:name"
	SignalSpans        map[string][]modelservice.Span // The matched spans of span rules keyed by "type:name"; set at the response stage
	SignalErrorMatches map[string]bool
	Diagnostics        decision.EvaluationDiagnostics

	// Signal metrics (only populated in eval mode)
	Metrics *SignalMetricsCollection

	// labelReferences are the "<type>:<rule>:<option>" references a rule a
	// condition can name by itself publishes for label conditions only.
	labelReferences map[string]bool
}

// Matches is the decision engine's view of the results: the matches, values
// and failures a rule tree reads.
func (r *SignalResults) Matches() *decision.SignalMatches {
	return &decision.SignalMatches{
		KeywordRules:       r.MatchedKeywordRules,
		DomainRules:        r.MatchedDomainRules,
		FactCheckRules:     r.MatchedFactCheckRules,
		UserFeedbackRules:  r.MatchedUserFeedbackRules,
		LanguageRules:      r.MatchedLanguageRules,
		ContextRules:       r.MatchedContextRules,
		ComplexityRules:    r.MatchedComplexityRules,
		ModalityRules:      r.MatchedModalityRules,
		SignalConfidences:  r.SignalConfidences,
		AuthzRules:         r.MatchedAuthzRules,
		JailbreakRules:     r.MatchedJailbreakRules,
		SafetyRules:        r.MatchedSafetyRules,
		PIIRules:           r.MatchedPIIRules,
		TurnRelationRules:  r.MatchedTurnRelationRules,
		ConversationRules:  r.MatchedConversationRules,
		MetadataRules:      r.MatchedMetadataRules,
		InputModalityRules: r.MatchedInputModalityRules,
		SystemOneRules:     r.MatchedSystemOneRules,
		ProjectionRules:    r.MatchedProjectionRules,
		SignalValues:       r.SignalValues,
		SignalErrors:       r.SignalErrors,
		SignalErrorMatches: r.SignalErrorMatches,
	}
}

// SignalMetricsCollection contains metrics for all signal types.
type SignalMetricsCollection struct {
	Keyword       SignalMetrics `json:"keyword"`
	Domain        SignalMetrics `json:"domain"`
	FactCheck     SignalMetrics `json:"fact_check"`
	UserFeedback  SignalMetrics `json:"user_feedback"`
	TurnRelation  SignalMetrics `json:"turn_relation"`
	Language      SignalMetrics `json:"language"`
	Context       SignalMetrics `json:"context"`
	Complexity    SignalMetrics `json:"complexity"`
	Modality      SignalMetrics `json:"modality"`
	Authz         SignalMetrics `json:"authz"`
	Jailbreak     SignalMetrics `json:"jailbreak"`
	Safety        SignalMetrics `json:"safety"`
	PII           SignalMetrics `json:"pii"`
	Conversation  SignalMetrics `json:"conversation"`
	Metadata      SignalMetrics `json:"metadata"`
	InputModality SignalMetrics `json:"input_modality"`
	SystemOne     SignalMetrics `json:"systemone"`
}

// appendMatched records a matched reference of a systemone or application
// signal type, which rule published.
func (r *SignalResults) appendMatched(signalType string, rule config.SystemOneRule, reference string) {
	matched := r.matchedRulesOf(signalType)
	if matched == nil {
		return
	}
	*matched = append(*matched, reference)
	if reference != rule.Name && rule.MatchesItself() {
		if r.labelReferences == nil {
			r.labelReferences = make(map[string]bool)
		}
		r.labelReferences[signalType+":"+reference] = true
	}
}

// ReportedMatches are a systemone or application signal type's matches as
// its matched-rule header and Router Replay list them: each rule that
// matched, by name, and for a choice rule without labels, which conditions
// name by label, the option its answer chose, as "<rule>:<option>". The label
// references a rule that conditions name by itself publishes for label
// conditions, a qualifying option of a rule that did not match included, are
// not listed.
func (r *SignalResults) ReportedMatches(signalType string) []string {
	matched := r.matchedRulesOf(signalType)
	if matched == nil {
		return nil
	}
	var reported []string
	for _, reference := range *matched {
		if !r.labelReferences[signalType+":"+reference] {
			reported = append(reported, reference)
		}
	}
	return reported
}

// matchedRulesOf is the matched-reference list of a systemone or application
// signal type; nil for a type asked of no System One model.
func (r *SignalResults) matchedRulesOf(signalType string) *[]string {
	switch signalType {
	case config.SignalTypeSystemOne:
		return &r.MatchedSystemOneRules
	case config.SignalTypeDomain:
		return &r.MatchedDomainRules
	case config.SignalTypeFactCheck:
		return &r.MatchedFactCheckRules
	case config.SignalTypeUserFeedback:
		return &r.MatchedUserFeedbackRules
	case config.SignalTypeJailbreak:
		return &r.MatchedJailbreakRules
	case config.SignalTypeSafety:
		return &r.MatchedSafetyRules
	case config.SignalTypePII:
		return &r.MatchedPIIRules
	case config.SignalTypeTurnRelation:
		return &r.MatchedTurnRelationRules
	case config.SignalTypeHallucination:
		return &r.MatchedHallucinationRules
	case config.SignalTypeRefusal:
		return &r.MatchedRefusalRules
	case config.SignalTypeComplexity:
		return &r.MatchedComplexityRules
	case config.SignalTypeModality:
		return &r.MatchedModalityRules
	}
	return nil
}

// metricsOf is the metrics entry of a systemone or application signal type
// asked at the request stage; nil for any other type.
func (m *SignalMetricsCollection) metricsOf(signalType string) *SignalMetrics {
	switch signalType {
	case config.SignalTypeSystemOne:
		return &m.SystemOne
	case config.SignalTypeDomain:
		return &m.Domain
	case config.SignalTypeFactCheck:
		return &m.FactCheck
	case config.SignalTypeUserFeedback:
		return &m.UserFeedback
	case config.SignalTypeJailbreak:
		return &m.Jailbreak
	case config.SignalTypeSafety:
		return &m.Safety
	case config.SignalTypePII:
		return &m.PII
	case config.SignalTypeTurnRelation:
		return &m.TurnRelation
	case config.SignalTypeComplexity:
		return &m.Complexity
	case config.SignalTypeModality:
		return &m.Modality
	}
	return nil
}
