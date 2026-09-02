package config

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

type Signals struct {
	KeywordRules       []KeywordRule          `yaml:"keyword_rules,omitempty"`
	EmbeddingRules     []EmbeddingRule        `yaml:"embedding_rules,omitempty"`
	Categories         []Category             `yaml:"categories"`
	FactCheckRules     []FactCheckRule        `yaml:"fact_check_rules,omitempty"`
	UserFeedbackRules  []UserFeedbackRule     `yaml:"user_feedback_rules,omitempty"`
	ReaskRules         []ReaskRule            `yaml:"reask_rules,omitempty"`
	PreferenceRules    []PreferenceRule       `yaml:"preference_rules,omitempty"`
	LanguageRules      []LanguageRule         `yaml:"language_rules,omitempty"`
	ContextRules       []ContextRule          `yaml:"context_rules,omitempty"`
	StructureRules     []StructureRule        `yaml:"structure_rules,omitempty"`
	ComplexityRules    []ComplexityRule       `yaml:"complexity_rules,omitempty"`
	ModalityRules      []ModalityRule         `yaml:"modality_rules,omitempty"`
	RoleBindings       []RoleBinding          `yaml:"role_bindings,omitempty"`
	JailbreakRules     []JailbreakRule        `yaml:"jailbreak,omitempty"`
	PIIRules           []PIIRule              `yaml:"pii,omitempty"`
	KBRules            []KBSignalRule         `yaml:"kb,omitempty"`
	ConversationRules  []ConversationRule     `yaml:"conversation,omitempty"`
	EventRules         []EventRule            `yaml:"events,omitempty"`
	MetadataRules      []MetadataRule         `yaml:"metadata,omitempty"`
	ClassifierRules    []ClassifierSignalRule `yaml:"classifiers,omitempty"`
	InputModalityRules []InputModalityRule    `yaml:"input_modality,omitempty"`
}

// EventRule matches structured event metadata extracted from request text.
// It routes event-driven requests (error alerts, audit logs, incident payloads)
// to specialized model pools based on event type, severity, and temporal urgency.
type EventRule struct {
	Name        string   `yaml:"name"`
	Description string   `yaml:"description,omitempty"`
	EventTypes  []string `yaml:"event_types,omitempty"`  // e.g. ["payment_failed", "auth_error"]
	Severities  []string `yaml:"severities,omitempty"`   // e.g. ["critical", "high"]
	ActionCodes []string `yaml:"action_codes,omitempty"` // domain-specific codes, e.g. ["TXN_DECLINE"]
	Temporal    bool     `yaml:"temporal,omitempty"`     // match time-sensitive markers (urgent, immediate)
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

// QueryModality declares which modality of incoming request payload the
// embedding rule's query is computed from. The candidates remain text in
// every case: the rule cosine-matches a text-anchor set against a query
// embedding from the declared modality, all in the shared multimodal space.
//
// "text"  (default, backward-compatible): query embedded from request text.
// "image": query embedded from an image attachment (base64, data-URI, or path).
// "audio": query embedded from an audio attachment (base64, data-URI, or path).
//
// "image" and "audio" require global.model_catalog.embeddings.semantic.embedding_config.model_type=multimodal.
type QueryModality string

const (
	QueryModalityText  QueryModality = "text"
	QueryModalityImage QueryModality = "image"
	QueryModalityAudio QueryModality = "audio"
)

type EmbeddingRule struct {
	Name                      string            `yaml:"name"`
	SimilarityThreshold       float32           `yaml:"threshold"`
	Candidates                []string          `yaml:"candidates"`
	AggregationMethodConfiged AggregationMethod `yaml:"aggregation_method"`
	// QueryModality controls which modality of the incoming request payload
	// the query embedding is computed from. Defaults to "text" when omitted,
	// preserving existing behavior.
	QueryModality QueryModality `yaml:"query_modality,omitempty"`
}

// EffectiveQueryModality returns the rule's declared query modality, or
// QueryModalityText when unset. Comparison should always go through this
// helper so default behavior stays consistent across call sites.
func (r EmbeddingRule) EffectiveQueryModality() QueryModality {
	m := QueryModality(strings.ToLower(strings.TrimSpace(string(r.QueryModality))))
	if m == "" {
		return QueryModalityText
	}
	return m
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
	// Direction is the stage the rule observes: "request" (the default) scores
	// the prompt before a model is selected, "response" scores the model's
	// output once it has answered. See JailbreakRule.Stage.
	Direction string `yaml:"direction,omitempty"`
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
	// Threshold is the minimum lingua-go confidence score required to accept a
	// language detection result for this rule. When unset (0), the classifier
	// uses its built-in default of 0.3. Setting a higher value (e.g. 0.6)
	// reduces false-positive language matches on short or ambiguous text.
	Threshold float32 `yaml:"threshold,omitempty"`
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
	if err != nil || math.IsNaN(val) || math.IsInf(val, 0) {
		return 0, fmt.Errorf("invalid token count format: %s", t)
	}
	if val < 0 {
		return 0, fmt.Errorf("token count must not be negative: %s", t)
	}
	scaled := val * multiplier
	if scaled >= float64(math.MaxInt) {
		return 0, fmt.Errorf("token count is too large: %s", t)
	}
	return int(scaled), nil
}

// IsSet reports whether the token count was configured (non-empty after trimming).
func (t TokenCount) IsSet() bool {
	return strings.TrimSpace(string(t)) != ""
}

// ContextRule matches a request whose estimated token count falls inside an
// inclusive band: min_tokens <= count <= max_tokens. A rule with
// min_tokens == max_tokens matches exactly one count. Omitting max_tokens
// makes the band open-ended: every count at or above min_tokens matches.
// Omitting min_tokens means 0.
type ContextRule struct {
	Name      string     `yaml:"name"`
	MinTokens TokenCount `yaml:"min_tokens"`
	// MaxTokens is the inclusive upper bound. Leave it empty for no upper bound.
	MaxTokens   TokenCount `yaml:"max_tokens,omitempty"`
	Description string     `yaml:"description,omitempty"`
}

// ContextBounds is the parsed form of a ContextRule band.
type ContextBounds struct {
	Min int
	Max int
	// Unbounded is true when max_tokens is omitted; Max is then meaningless.
	Unbounded bool
}

// Matches reports whether count falls inside the band.
func (b ContextBounds) Matches(count int) bool {
	return count >= b.Min && (b.Unbounded || count <= b.Max)
}

// Overlaps reports whether the two inclusive bands share at least one count.
func (b ContextBounds) Overlaps(other ContextBounds) bool {
	if !b.Unbounded && b.Max < other.Min {
		return false
	}
	if !other.Unbounded && other.Max < b.Min {
		return false
	}
	return true
}

// NamedContextBand pairs a context rule name with its parsed bounds.
type NamedContextBand struct {
	Name   string
	Bounds ContextBounds
}

// ContextBandOverlap records two bands that share at least one token count.
// Contains is true when Outer fully covers Inner, which is a common and
// usually intentional layout (a broad band plus narrower specialisations).
type ContextBandOverlap struct {
	Outer    NamedContextBand
	Inner    NamedContextBand
	Contains bool
}

// ContextBandGap records a run of token counts covered by no band. Before is
// the band that starts right after the gap.
type ContextBandGap struct {
	From   int
	To     int
	Before NamedContextBand
}

// ContextBandIssues is the shared band analysis behind both the YAML and DSL
// validators. It returns overlaps between bands and gaps below a band's
// minimum that no earlier band covers. Counts below the lowest band are not
// reported as a gap. The input is not modified.
func ContextBandIssues(bands []NamedContextBand) (overlaps []ContextBandOverlap, gaps []ContextBandGap) {
	sorted := append([]NamedContextBand(nil), bands...)
	sort.SliceStable(sorted, func(i, j int) bool {
		return sorted[i].Bounds.Min < sorted[j].Bounds.Min
	})

	for i := range sorted {
		for j := i + 1; j < len(sorted); j++ {
			if !sorted[i].Bounds.Overlaps(sorted[j].Bounds) {
				continue
			}
			overlaps = append(overlaps, ContextBandOverlap{
				Outer:    sorted[i],
				Inner:    sorted[j],
				Contains: sorted[i].Bounds.contains(sorted[j].Bounds),
			})
		}
	}

	coveredTo := -1
	for _, band := range sorted {
		if coveredTo >= 0 && band.Bounds.Min > coveredTo+1 {
			gaps = append(gaps, ContextBandGap{From: coveredTo + 1, To: band.Bounds.Min - 1, Before: band})
		}
		if band.Bounds.Unbounded {
			break
		}
		if band.Bounds.Max > coveredTo {
			coveredTo = band.Bounds.Max
		}
	}
	return overlaps, gaps
}

// contains reports whether b fully covers other. Callers pass bands sorted by
// Min, so only the upper edge needs checking beyond the Min comparison.
func (b ContextBounds) contains(other ContextBounds) bool {
	if b.Min > other.Min {
		return false
	}
	if b.Unbounded {
		return true
	}
	return !other.Unbounded && other.Max <= b.Max
}

// String renders the band for diagnostics, e.g. "[0, 1000]" or "[8000, ∞)".
func (b ContextBounds) String() string {
	if b.Unbounded {
		return fmt.Sprintf("[%d, ∞)", b.Min)
	}
	return fmt.Sprintf("[%d, %d]", b.Min, b.Max)
}

// Bounds parses the rule's token limits. A missing min_tokens defaults to 0.
// It returns an error when neither limit is set, either value fails to parse,
// or min_tokens exceeds max_tokens.
func (r ContextRule) Bounds() (ContextBounds, error) {
	if !r.MinTokens.IsSet() && !r.MaxTokens.IsSet() {
		return ContextBounds{}, fmt.Errorf("min_tokens or max_tokens must be set")
	}
	minTokens, err := r.MinTokens.Value()
	if err != nil {
		return ContextBounds{}, fmt.Errorf("min_tokens: %w", err)
	}
	if !r.MaxTokens.IsSet() {
		return ContextBounds{Min: minTokens, Unbounded: true}, nil
	}
	maxTokens, err := r.MaxTokens.Value()
	if err != nil {
		return ContextBounds{}, fmt.Errorf("max_tokens: %w", err)
	}
	if minTokens > maxTokens {
		return ContextBounds{}, fmt.Errorf(
			"min_tokens (%s) must not exceed max_tokens (%s); use equal values for an exact match or omit max_tokens for no upper bound",
			strings.TrimSpace(string(r.MinTokens)), strings.TrimSpace(string(r.MaxTokens)),
		)
	}
	return ContextBounds{Min: minTokens, Max: maxTokens}, nil
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

type CategoryMetadata struct {
	Name           string   `yaml:"name"`
	Description    string   `yaml:"description,omitempty"`
	MMLUCategories []string `yaml:"mmlu_categories,omitempty"`
}
