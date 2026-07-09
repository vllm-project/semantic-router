package config

// Decision represents a routing decision that combines multiple rules with boolean logic.
type Decision struct {
	Name                string                     `yaml:"name"`
	Description         string                     `yaml:"description,omitempty"`
	Priority            int                        `yaml:"priority,omitempty"`
	Tier                int                        `yaml:"tier,omitempty"`
	OutputContract      string                     `yaml:"output_contract,omitempty" json:"output_contract,omitempty"`
	OutputContractSpec  *OutputContractSpec        `yaml:"output_contract_spec,omitempty" json:"output_contract_spec,omitempty"`
	Rules               RuleCombination            `yaml:"rules"`
	ModelRefs           []ModelRef                 `yaml:"modelRefs,omitempty"`
	Algorithm           *AlgorithmConfig           `yaml:"algorithm,omitempty"`
	Adaptations         DecisionAdaptationsConfig  `yaml:"adaptations,omitempty"`
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
// Runtime consumes Drop (semantic-cache write skip), TTLTurns (per-entry
// cache TTL override), and KeepCurrentModel (model-switch-gate forced stay).
// PreferPrefixRetention is emitted to the pool as an x-vsr-retention-prefer-prefix
// header; its session-aware scoring bias and KV-cache eviction integration are
// follow-up work. All set fields are also observed via log + trace attributes
// and emitted as x-vsr-retention-* response headers (issues #1747, #2009).
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
	Fusion       *FusionAlgorithmConfig       `yaml:"fusion,omitempty"`
	Workflows    *WorkflowsAlgorithmConfig    `yaml:"workflows,omitempty"`
	Elo          *EloSelectionConfig          `yaml:"-"`
	RouterDC     *RouterDCSelectionConfig     `yaml:"router_dc,omitempty"`
	AutoMix      *AutoMixSelectionConfig      `yaml:"automix,omitempty"`
	Hybrid       *HybridSelectionConfig       `yaml:"hybrid,omitempty"`
	RLDriven     *RLDrivenSelectionConfig     `yaml:"-"`
	GMTRouter    *GMTRouterSelectionConfig    `yaml:"-"`
	LatencyAware *LatencyAwareAlgorithmConfig `yaml:"latency_aware,omitempty"`
	MultiFactor  *MultiFactorSelectionConfig  `yaml:"multi_factor,omitempty"`
	SessionAware *SessionAwareSelectionConfig `yaml:"-"`
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

	// VerifierServerURL points to the AutoMix self-verification server when
	// confidence_method=automix_entailment. The server implements few-shot
	// entailment verification per arXiv:2310.12963 §3.2 and is reached over
	// HTTP via selection.AutoMixVerifierClient. A reference implementation
	// lives at src/training/model_selection/rl_model_selection/automix_verifier.py.
	// Required only when confidence_method=automix_entailment; ignored otherwise.
	VerifierServerURL string `yaml:"verifier_server_url,omitempty"`

	// VerifierTimeoutSeconds bounds each verifier HTTP call. Defaults to 60
	// when zero, matching selection.NewAutoMixVerifierClient. Only consulted
	// when confidence_method=automix_entailment.
	VerifierTimeoutSeconds int `yaml:"verifier_timeout_seconds,omitempty"`
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
	SynthesisModel               string  `yaml:"synthesis_model,omitempty"`
	MaxConcurrent                int     `yaml:"max_concurrent,omitempty"`
	RoundTimeoutSeconds          int     `yaml:"round_timeout_seconds,omitempty"`
	MinSuccessfulResponses       int     `yaml:"min_successful_responses,omitempty"`
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
