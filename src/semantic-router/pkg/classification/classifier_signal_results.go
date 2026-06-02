package classification

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/projectiontrace"

// SignalMetrics contains performance and probability metrics for a single signal.
type SignalMetrics struct {
	ExecutionTimeMs float64 `json:"execution_time_ms"` // Execution time in milliseconds
	Confidence      float64 `json:"confidence"`        // Confidence score (0.0-1.0), 0 if not applicable
}

// SignalResults contains all evaluated signal results.
type SignalResults struct {
	MatchedKeywordRules      []string
	MatchedKeywords          []string // The actual keywords that matched (not rule names)
	MatchedEmbeddingRules    []string
	MatchedDomainRules       []string
	MatchedFactCheckRules    []string // "needs_fact_check" or "no_fact_check_needed"
	MatchedUserFeedbackRules []string // "satisfied", "need_clarification", "wrong_answer", "want_different"
	MatchedReaskRules        []string // History-aware repeated-question dissatisfaction signals
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
	MatchedConversationRules []string
	MatchedEventRules        []string // Matched event rule names (event type, severity, temporal, action codes)
	MatchedProjectionRules   []string // Matched derived routing outputs from routing.projections.mappings
	ProjectionScores         map[string]float64
	ProjectionTrace          *projectiontrace.Trace // Explainability payload for projections (replay / dashboard)

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

// SignalMetricsCollection contains metrics for all signal types.
type SignalMetricsCollection struct {
	Keyword      SignalMetrics `json:"keyword"`
	Embedding    SignalMetrics `json:"embedding"`
	Domain       SignalMetrics `json:"domain"`
	FactCheck    SignalMetrics `json:"fact_check"`
	UserFeedback SignalMetrics `json:"user_feedback"`
	Reask        SignalMetrics `json:"reask"`
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
	Conversation SignalMetrics `json:"conversation"`
	Event        SignalMetrics `json:"event"`
}
