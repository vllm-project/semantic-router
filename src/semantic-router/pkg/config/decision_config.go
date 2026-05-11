package config

// Decision represents a routing decision that combines multiple rules with boolean logic.
type Decision struct {
	Name                string                     `yaml:"name"`
	Description         string                     `yaml:"description,omitempty"`
	Priority            int                        `yaml:"priority,omitempty"`
	Tier                int                        `yaml:"tier,omitempty"`
	Rules               RuleCombination            `yaml:"rules"`
	ModelRefs           []ModelRef                 `yaml:"modelRefs,omitempty"`
	Algorithm           *AlgorithmConfig           `yaml:"algorithm,omitempty"`
	Plugins             []DecisionPlugin           `yaml:"plugins,omitempty"`
	CandidateIterations []CandidateIterationConfig `yaml:"candidateIterations,omitempty"`
	// Emits carries declarative side-effect directives produced by EMIT blocks
	// inside the matching decision branch. The slice preserves DSL declaration
	// order so round-trip decompilation stays stable.
	Emits []EmitDirective `yaml:"emits,omitempty" json:"emits,omitempty"`
}

// EmitDirective is a tagged-union wrapper for declarative directives emitted
// by a decision branch. The initial supported Kind is "retention".
type EmitDirective struct {
	Kind      string              `yaml:"kind" json:"kind"`
	Retention *RetentionDirective `yaml:"retention,omitempty" json:"retention,omitempty"`
}

// RetentionDirective expresses keep / drop / prefer-retain semantics over the
// response/cache surface. All fields are tri-state pointers so we can
// distinguish "unset" from an explicit zero value.
//
// MVP runtime consumes only Drop. The remaining fields are contract-only and
// observed via log + trace attributes; their runtime consumers will land in
// follow-up PRs (#1742 turn telemetry, #1749 model_switch_gate, prefix cache
// eviction hints).
type RetentionDirective struct {
	Drop                  *bool `yaml:"drop,omitempty" json:"drop,omitempty"`
	TTLTurns              *int  `yaml:"ttl_turns,omitempty" json:"ttl_turns,omitempty"`
	KeepCurrentModel      *bool `yaml:"keep_current_model,omitempty" json:"keep_current_model,omitempty"`
	PreferPrefixRetention *bool `yaml:"prefer_prefix_retention,omitempty" json:"prefer_prefix_retention,omitempty"`
}

// CandidateIterationConfig is the canonical, bounded representation of a DSL
// FOR ... IN block over candidate models. It is declarative metadata consumed
// by selection/policy layers, not a runtime script.
type CandidateIterationConfig struct {
	Variable string                           `yaml:"variable"`
	Source   string                           `yaml:"source"`
	Models   []ModelRef                       `yaml:"models,omitempty"`
	Outputs  []CandidateIterationOutputConfig `yaml:"outputs,omitempty"`
}

// CandidateIterationOutputConfig describes how an iteration feeds an existing
// routing construct. The initial supported output is type=model, value=<var>.
type CandidateIterationOutputConfig struct {
	Type  string `yaml:"type"`
	Value string `yaml:"value,omitempty"`
}

// AlgorithmConfig defines how multiple models should be executed and aggregated.
type AlgorithmConfig struct {
	Type         string                       `yaml:"type"`
	Confidence   *ConfidenceAlgorithmConfig   `yaml:"confidence,omitempty"`
	Ratings      *RatingsAlgorithmConfig      `yaml:"ratings,omitempty"`
	ReMoM        *ReMoMAlgorithmConfig        `yaml:"remom,omitempty"`
	Elo          *EloSelectionConfig          `yaml:"elo,omitempty"`
	RouterDC     *RouterDCSelectionConfig     `yaml:"router_dc,omitempty"`
	AutoMix      *AutoMixSelectionConfig      `yaml:"automix,omitempty"`
	Hybrid       *HybridSelectionConfig       `yaml:"hybrid,omitempty"`
	RLDriven     *RLDrivenSelectionConfig     `yaml:"rl_driven,omitempty"`
	GMTRouter    *GMTRouterSelectionConfig    `yaml:"gmtrouter,omitempty"`
	LatencyAware *LatencyAwareAlgorithmConfig `yaml:"latency_aware,omitempty"`
	OnError      string                       `yaml:"on_error,omitempty"`
}

type ConfidenceAlgorithmConfig struct {
	ConfidenceMethod    string               `yaml:"confidence_method,omitempty"`
	Threshold           float64              `yaml:"threshold,omitempty"`
	HybridWeights       *HybridWeightsConfig `yaml:"hybrid_weights,omitempty"`
	OnError             string               `yaml:"on_error,omitempty"`
	EscalationOrder     string               `yaml:"escalation_order,omitempty"`
	CostQualityTradeoff float64              `yaml:"cost_quality_tradeoff,omitempty"`
	TokenFilter         string               `yaml:"token_filter,omitempty"`
}

type HybridWeightsConfig struct {
	LogprobWeight float64 `yaml:"logprob_weight,omitempty"`
	MarginWeight  float64 `yaml:"margin_weight,omitempty"`
}

type RatingsAlgorithmConfig struct {
	MaxConcurrent int    `yaml:"max_concurrent,omitempty"`
	OnError       string `yaml:"on_error,omitempty"`
}

type ReMoMAlgorithmConfig struct {
	BreadthSchedule              []int   `yaml:"breadth_schedule"`
	ModelDistribution            string  `yaml:"model_distribution,omitempty"`
	Temperature                  float64 `yaml:"temperature,omitempty"`
	IncludeReasoning             bool    `yaml:"include_reasoning,omitempty"`
	CompactionStrategy           string  `yaml:"compaction_strategy,omitempty"`
	CompactionTokens             int     `yaml:"compaction_tokens,omitempty"`
	SynthesisTemplate            string  `yaml:"synthesis_template,omitempty"`
	MaxConcurrent                int     `yaml:"max_concurrent,omitempty"`
	OnError                      string  `yaml:"on_error,omitempty"`
	ShuffleSeed                  int     `yaml:"shuffle_seed,omitempty"`
	IncludeIntermediateResponses bool    `yaml:"include_intermediate_responses,omitempty"`
	MaxResponsesPerRound         int     `yaml:"max_responses_per_round,omitempty"`
}

type ModelReasoningControl struct {
	UseReasoning         *bool  `yaml:"use_reasoning"`
	ReasoningDescription string `yaml:"reasoning_description,omitempty"`
	ReasoningEffort      string `yaml:"reasoning_effort,omitempty"`
}

type ModelRef struct {
	Model                 string  `yaml:"model"`
	LoRAName              string  `yaml:"lora_name,omitempty"`
	Weight                float64 `yaml:"weight,omitempty"`
	ModelReasoningControl `yaml:",inline"`
}

// RuleNode is a recursive boolean expression tree over signal references.
type RuleNode struct {
	Type       string     `yaml:"type,omitempty" json:"type,omitempty"`
	Name       string     `yaml:"name,omitempty" json:"name,omitempty"`
	Operator   string     `yaml:"operator,omitempty" json:"operator,omitempty"`
	Conditions []RuleNode `yaml:"conditions,omitempty" json:"conditions,omitempty"`
}

func (n *RuleNode) IsLeaf() bool {
	return n.Type != ""
}

type (
	RuleCombination = RuleNode
	RuleCondition   = RuleNode
)
