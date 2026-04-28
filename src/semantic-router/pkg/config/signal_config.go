package config

import (
	"fmt"
	"strconv"
	"strings"
)

type Signals struct {
	KeywordRules       []KeywordRule       `yaml:"keyword_rules,omitempty"`
	EmbeddingRules     []EmbeddingRule     `yaml:"embedding_rules,omitempty"`
	Categories         []Category          `yaml:"categories"`
	FactCheckRules     []FactCheckRule     `yaml:"fact_check_rules,omitempty"`
	UserFeedbackRules  []UserFeedbackRule  `yaml:"user_feedback_rules,omitempty"`
	ReaskRules         []ReaskRule         `yaml:"reask_rules,omitempty"`
	PreferenceRules    []PreferenceRule    `yaml:"preference_rules,omitempty"`
	LanguageRules      []LanguageRule      `yaml:"language_rules,omitempty"`
	ContextRules       []ContextRule       `yaml:"context_rules,omitempty"`
	StructureRules     []StructureRule     `yaml:"structure_rules,omitempty"`
	ComplexityRules    []ComplexityRule    `yaml:"complexity_rules,omitempty"`
	ModalityRules      []ModalityRule      `yaml:"modality_rules,omitempty"`
	RoleBindings       []RoleBinding       `yaml:"role_bindings,omitempty"`
	JailbreakRules     []JailbreakRule     `yaml:"jailbreak,omitempty"`
	PIIRules           []PIIRule           `yaml:"pii,omitempty"`
	KBRules            []KBSignalRule      `yaml:"kb,omitempty"`
	ConversationRules  []ConversationRule  `yaml:"conversation,omitempty"`
	SessionMetricRules  []SessionMetricRule  `yaml:"session_metrics,omitempty"`
	EventContextRules   []EventContextRule   `yaml:"event_context_rules,omitempty"`
}

// SessionMetricRule is a unified session-context routing metric.
// Use kind "state" for SESSION_STATE scalars (with optional normalize), or kind "lookup"
// for table-backed values resolved via LookupResolver at request time.
type SessionMetricRule struct {
	Name      string   `yaml:"name"`
	Kind      string   `yaml:"kind"` // "state" or "lookup"
	State     string   `yaml:"state,omitempty"`
	Normalize string   `yaml:"normalize,omitempty"`
	Min       *float64 `yaml:"min,omitempty"`
	Max       *float64 `yaml:"max,omitempty"`
	Table     string   `yaml:"table,omitempty"`
	Key       []string `yaml:"key,omitempty"`
}

// EventContextRule matches structured event metadata extracted from request text.
// It routes event-driven requests (error alerts, audit logs, incident payloads)
// to specialized model pools based on event type, severity, and temporal urgency.
type EventContextRule struct {
	Name        string   `yaml:"name"`
	EventTypes  []string `yaml:"event_types,omitempty"`   // e.g. ["payment_failed", "auth_error"]
	Severities  []string `yaml:"severities,omitempty"`    // e.g. ["critical", "high"]
	ActionCodes []string `yaml:"action_codes,omitempty"`  // domain-specific codes, e.g. ["TXN_DECLINE"]
	Temporal    bool     `yaml:"temporal,omitempty"`      // match time-sensitive markers (urgent, immediate)
}

type KeywordRule struct {
	Name           string   `yaml:"name"`
	Operator       string   `yaml:"operator"`
	Keywords       []string `yaml:"keywords"`
	CaseSensitive  bool     `yaml:"case_sensitive"`
	Method         string   `yaml:"method,omitempty"`
	FuzzyMatch     bool     `yaml:"fuzzy_match,omitempty"`
	FuzzyThreshold int      `yaml:"fuzzy_threshold,omitempty"`
	BM25Threshold  float32  `yaml:"bm25_threshold,omitempty"`
	NgramThreshold float32  `yaml:"ngram_threshold,omitempty"`
	NgramArity     int      `yaml:"ngram_arity,omitempty"`
}

type AggregationMethod string

const (
	AggregationMethodMean AggregationMethod = "mean"
	AggregationMethodMax  AggregationMethod = "max"
	AggregationMethodAny  AggregationMethod = "any"
)

type EmbeddingRule struct {
	Name                      string            `yaml:"name"`
	SimilarityThreshold       float32           `yaml:"threshold"`
	Candidates                []string          `yaml:"candidates"`
	AggregationMethodConfiged AggregationMethod `yaml:"aggregation_method"`
}

type FactCheckRule struct {
	Name        string `yaml:"name"`
	Description string `yaml:"description,omitempty"`
}

type UserFeedbackRule struct {
	Name        string `yaml:"name"`
	Description string `yaml:"description,omitempty"`
}

type ReaskRule struct {
	Name          string  `yaml:"name"`
	Description   string  `yaml:"description,omitempty"`
	Threshold     float32 `yaml:"threshold,omitempty"`
	LookbackTurns int     `yaml:"lookback_turns,omitempty"`
}

func (r ReaskRule) WithDefaults() ReaskRule {
	result := r
	if result.Threshold == 0 {
		result.Threshold = 0.8
	}
	if result.LookbackTurns == 0 {
		result.LookbackTurns = 1
	}
	return result
}

type ModalityRule struct {
	Name        string `yaml:"name"`
	Description string `yaml:"description,omitempty"`
}

type JailbreakRule struct {
	Name              string   `yaml:"name"`
	Method            string   `yaml:"method,omitempty"`
	Threshold         float32  `yaml:"threshold"`
	IncludeHistory    bool     `yaml:"include_history,omitempty"`
	Description       string   `yaml:"description,omitempty"`
	JailbreakPatterns []string `yaml:"jailbreak_patterns,omitempty"`
	BenignPatterns    []string `yaml:"benign_patterns,omitempty"`
}

type PIIRule struct {
	Name            string   `yaml:"name"`
	Threshold       float32  `yaml:"threshold"`
	PIITypesAllowed []string `yaml:"pii_types_allowed,omitempty"`
	IncludeHistory  bool     `yaml:"include_history,omitempty"`
	Description     string   `yaml:"description,omitempty"`
}

type PreferenceRule struct {
	Name        string   `yaml:"name"`
	Description string   `yaml:"description,omitempty"`
	Examples    []string `yaml:"examples,omitempty"`
	Threshold   float32  `yaml:"threshold,omitempty"`
}

type LanguageRule struct {
	Name        string `yaml:"name"`
	Description string `yaml:"description,omitempty"`
}

type TokenCount string

func (t TokenCount) Value() (int, error) {
	s := strings.ToUpper(strings.TrimSpace(string(t)))
	if s == "" {
		return 0, nil
	}

	multiplier := 1.0
	if strings.HasSuffix(s, "K") {
		multiplier = 1000.0
		s = strings.TrimSuffix(s, "K")
	} else if strings.HasSuffix(s, "M") {
		multiplier = 1000000.0
		s = strings.TrimSuffix(s, "M")
	}

	val, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, fmt.Errorf("invalid token count format: %s", t)
	}
	return int(val * multiplier), nil
}

type ContextRule struct {
	Name        string     `yaml:"name"`
	MinTokens   TokenCount `yaml:"min_tokens"`
	MaxTokens   TokenCount `yaml:"max_tokens"`
	Description string     `yaml:"description,omitempty"`
}

type StructureRule struct {
	Name        string            `yaml:"name"`
	Description string            `yaml:"description,omitempty"`
	Feature     StructureFeature  `yaml:"feature"`
	Predicate   *NumericPredicate `yaml:"predicate,omitempty"`
}

type StructureFeature struct {
	Type   string          `yaml:"type"`
	Source StructureSource `yaml:"source"`
}

type StructureSource struct {
	Type          string     `yaml:"type"`
	Pattern       string     `yaml:"pattern,omitempty"`
	Keywords      []string   `yaml:"keywords,omitempty"`
	CaseSensitive bool       `yaml:"case_sensitive,omitempty"`
	Sequences     [][]string `yaml:"sequences,omitempty"`
}

type ConversationRule struct {
	Name        string              `yaml:"name"`
	Description string              `yaml:"description,omitempty"`
	Feature     ConversationFeature `yaml:"feature"`
	Predicate   *NumericPredicate   `yaml:"predicate,omitempty"`
}

type ConversationFeature struct {
	Type   string             `yaml:"type"`
	Source ConversationSource `yaml:"source"`
}

type ConversationSource struct {
	Type string `yaml:"type"`
	Role string `yaml:"role,omitempty"`
}

type NumericPredicate struct {
	GT  *float64 `yaml:"gt,omitempty"`
	GTE *float64 `yaml:"gte,omitempty"`
	LT  *float64 `yaml:"lt,omitempty"`
	LTE *float64 `yaml:"lte,omitempty"`
}

type Subject struct {
	Kind string `yaml:"kind"`
	Name string `yaml:"name"`
}

type RoleBinding struct {
	Name        string    `yaml:"name"`
	Description string    `yaml:"description,omitempty"`
	Subjects    []Subject `yaml:"subjects"`
	Role        string    `yaml:"role"`
}

func (s *Signals) GetRoleBindings() []RoleBinding {
	return s.RoleBindings
}

type ComplexityCandidates struct {
	Candidates      []string `yaml:"candidates"`
	ImageCandidates []string `yaml:"image_candidates,omitempty"`
}

func HasImageCandidatesInRules(rules []ComplexityRule) bool {
	for _, r := range rules {
		if len(r.Hard.ImageCandidates) > 0 || len(r.Easy.ImageCandidates) > 0 {
			return true
		}
	}
	return false
}

type ComplexityRule struct {
	Name        string               `yaml:"name"`
	Threshold   float32              `yaml:"threshold"`
	Hard        ComplexityCandidates `yaml:"hard"`
	Easy        ComplexityCandidates `yaml:"easy"`
	Description string               `yaml:"description,omitempty"`
	Composer    *RuleCombination     `yaml:"composer,omitempty"`
}

type Category struct {
	CategoryMetadata `yaml:",inline"`
	ModelScores      []ModelScore `yaml:"model_scores,omitempty"`
}

type ModelScore struct {
	Model        string  `yaml:"model"`
	Score        float64 `yaml:"score"`
	UseReasoning *bool   `yaml:"use_reasoning"`
}

type DomainAwarePolicies struct {
	SystemPromptPolicy    `yaml:",inline"`
	SemanticCachingPolicy `yaml:",inline"`
	JailbreakPolicy       `yaml:",inline"`
	PIIDetectionPolicy    `yaml:",inline"`
}

type CategoryMetadata struct {
	Name           string   `yaml:"name"`
	Description    string   `yaml:"description,omitempty"`
	MMLUCategories []string `yaml:"mmlu_categories,omitempty"`
}

type SystemPromptPolicy struct {
	SystemPrompt        string `yaml:"system_prompt,omitempty"`
	SystemPromptEnabled *bool  `yaml:"system_prompt_enabled,omitempty"`
	SystemPromptMode    string `yaml:"system_prompt_mode,omitempty"`
}

type SemanticCachingPolicy struct {
	SemanticCacheEnabled             *bool    `yaml:"semantic_cache_enabled,omitempty"`
	SemanticCacheSimilarityThreshold *float32 `yaml:"semantic_cache_similarity_threshold,omitempty"`
}

type JailbreakPolicy struct {
	JailbreakEnabled   *bool    `yaml:"jailbreak_enabled,omitempty"`
	JailbreakThreshold *float32 `yaml:"jailbreak_threshold,omitempty"`
}

type PIIDetectionPolicy struct {
	PIIEnabled   *bool    `yaml:"pii_enabled,omitempty"`
	PIIThreshold *float32 `yaml:"pii_threshold,omitempty"`
}
