package config

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"

	"gopkg.in/yaml.v2"
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
	SafetyRules        []SafetyRule           `yaml:"safety,omitempty"`
	HallucinationRules []HallucinationRule    `yaml:"hallucination,omitempty"`
	PIIRules           []PIIRule              `yaml:"pii,omitempty"`
	KBRules            []KBSignalRule         `yaml:"kb,omitempty"`
	ConversationRules  []ConversationRule     `yaml:"conversation,omitempty"`
	EventRules         []EventRule            `yaml:"events,omitempty"`
	MetadataRules      []MetadataRule         `yaml:"metadata,omitempty"`
	ClassifierRules    []ClassifierSignalRule `yaml:"classifiers,omitempty"`
	InputModalityRules []InputModalityRule    `yaml:"input_modality,omitempty"`
}

// HallucinationRule declares the response-stage hallucination observation:
// the model's answer is checked against the grounding context the request
// carried (tool results or retrieved context). It has no request direction; it
// only exists once the model has answered, and the hallucination plugin of the
// decision selected for the request consumes it. The detector's own threshold
// and span filters stay on hallucination_model; a rule names the observation
// and chooses how it is explained.
type HallucinationRule struct {
	Name        string `yaml:"name"`
	Description string `yaml:"description,omitempty"`
	// UseNLI asks the detector for span-level NLI explanations. It decides how
	// the observation is produced, so it lives on the rule; the plugin's
	// use_nli is ignored once a rule is declared.
	UseNLI bool `yaml:"use_nli,omitempty"`
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
	Name                string  `yaml:"name"`
	SimilarityThreshold float32 `yaml:"threshold"`
	// Candidates and ImageCandidates form the positive bank. Negative candidates
	// optionally define a contrastive bank; the score is positive minus negative.
	Candidates                []string          `yaml:"candidates,omitempty"`
	ImageCandidates           []string          `yaml:"image_candidates,omitempty"`
	NegativeCandidates        []string          `yaml:"negative_candidates,omitempty"`
	NegativeImageCandidates   []string          `yaml:"negative_image_candidates,omitempty"`
	AggregationMethodConfiged AggregationMethod `yaml:"aggregation_method"`
	// QueryModality controls which modality of the incoming request payload
	// the query embedding is computed from. Defaults to "text" when omitted,
	// preserving existing behavior.
	QueryModality QueryModality `yaml:"query_modality,omitempty"`
	// PrototypeScoring overrides construction and scoring for this rule.
	// Text-only rules inherit the family config. Media queries or image
	// candidates retain all anchors and use raw max by default; a present object explicitly opts
	// into its complete policy (with built-in defaults).
	PrototypeScoring *PrototypeScoringConfig `yaml:"prototype_scoring,omitempty"`
}

func (r EmbeddingRule) HasImageCandidates() bool {
	return len(r.ImageCandidates) > 0 || len(r.NegativeImageCandidates) > 0
}

func (r EmbeddingRule) HasNegativeCandidates() bool {
	return len(r.NegativeCandidates) > 0 || len(r.NegativeImageCandidates) > 0
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

// EffectivePrototypeScoring preserves cross-modal anchors unless the rule
// explicitly opts into compression. Nearby text embeddings need not be
// interchangeable for an image or audio query.
func (r EmbeddingRule) EffectivePrototypeScoring(family PrototypeScoringConfig) PrototypeScoringConfig {
	if r.PrototypeScoring != nil {
		return r.PrototypeScoring.WithDefaults()
	}
	if r.HasImageCandidates() {
		disabled := false
		return PrototypeScoringConfig{Enabled: &disabled, BestWeight: 1, TopM: 1}.WithDefaults()
	}
	switch r.EffectiveQueryModality() {
	case QueryModalityImage, QueryModalityAudio:
		disabled := false
		return PrototypeScoringConfig{Enabled: &disabled, BestWeight: 1, TopM: 1}.WithDefaults()
	default:
		return family.WithDefaults()
	}
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
	Name string `yaml:"name"`
	// PrototypeScoring applies to all local text/image hard/easy banks. Nil
	// inherits the family config; a present object is a complete override.
	PrototypeScoring *PrototypeScoringConfig `yaml:"prototype_scoring,omitempty"`
	// Threshold is the symmetric shorthand, kept because the local margin is
	// signed and centred on zero: hard above +threshold, easy below
	// -threshold. Mutually exclusive with the explicit pair below.
	//
	// Zero and omitted mean the same thing to the local path, so a zero is
	// not written back out: a rule that states a pair must not grow a
	// `threshold: 0` on its way through the operator or the DSL emitter and
	// then be refused for stating both.
	Threshold float32              `yaml:"threshold,omitempty"`
	Hard      ComplexityCandidates `yaml:"hard"`
	Easy      ComplexityCandidates `yaml:"easy"`
	// The explicit boundary pair, for a score whose scale is the model's own
	// rather than a signed margin. The pair used states which way difficulty
	// runs, so no separate direction field is needed: hard_above with
	// easy_below where a higher score is harder, hard_below with easy_above
	// where a lower one is. Resolved by EffectiveBoundaries.
	HardAbove   *float64         `yaml:"hard_above,omitempty"`
	EasyBelow   *float64         `yaml:"easy_below,omitempty"`
	HardBelow   *float64         `yaml:"hard_below,omitempty"`
	EasyAbove   *float64         `yaml:"easy_above,omitempty"`
	Description string           `yaml:"description,omitempty"`
	Composer    *RuleCombination `yaml:"composer,omitempty"`

	// ThresholdSet records that the threshold key was written, so that a
	// written `threshold: 0` alongside an explicit pair can be refused the
	// way the CRD refuses it, instead of passing as an absent key. Threshold
	// is a float32 with no presence of its own, and widening it to a pointer
	// would ripple through the DSL compiler, the operator and every literal
	// that builds a rule. Never serialised: presence is a property of the
	// document the rule came from, not of the rule.
	ThresholdSet bool `yaml:"-" json:"-"`
}

// UnmarshalYAML decodes a rule and records whether `threshold` was written.
// The value is decoded exactly as before; only the presence is added.
func (r *ComplexityRule) UnmarshalYAML(unmarshal func(interface{}) error) error {
	type plain ComplexityRule
	var decoded plain
	if err := unmarshal(&decoded); err != nil {
		return err
	}
	// A second pass into a map is the presence probe. A map rather than a
	// one-field struct, because the strict loader refuses fields a struct
	// does not name. A null value counts as absent, as it does for the CRD.
	var raw map[interface{}]interface{}
	if err := unmarshal(&raw); err != nil {
		return err
	}
	*r = ComplexityRule(decoded)
	value, written := raw["threshold"]
	r.ThresholdSet = written && value != nil
	return nil
}

// MarshalYAML writes the rule as its fields, with one exception. Threshold
// carries omitempty so a pair rule does not grow a `threshold: 0` on its way
// through the operator or the DSL emitter, but a zero that was written has to
// survive emission: dropping it would let a rule the loader refuses pass once
// it has been serialised and read back. That one case is written through a
// map with the key restored. Every other rule is written exactly as before.
func (r ComplexityRule) MarshalYAML() (interface{}, error) {
	type plain ComplexityRule
	if !r.ThresholdSet || r.Threshold != 0 {
		return plain(r), nil
	}
	encoded, err := yaml.Marshal(plain(r))
	if err != nil {
		return nil, err
	}
	fields := map[string]interface{}{}
	if err := yaml.Unmarshal(encoded, &fields); err != nil {
		return nil, err
	}
	fields["threshold"] = r.Threshold
	return fields, nil
}

// thresholdDeclared reports whether the rule states a threshold at all: a
// written key, whatever its value, or a non-zero value from a caller that
// assigned the field directly.
func (r ComplexityRule) thresholdDeclared() bool {
	return r.ThresholdSet || r.Threshold != 0
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
