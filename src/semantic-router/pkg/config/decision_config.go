package config

import (
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
)

type UnknownPolicy string

const (
	RuleOnUnknownNoMatch     UnknownPolicy = "no_match"
	RuleOnUnknownMatch       UnknownPolicy = "match"
	RuleOnUnknownFailRequest UnknownPolicy = "fail_request"
)

var UnknownPolicies = []UnknownPolicy{RuleOnUnknownNoMatch, RuleOnUnknownMatch, RuleOnUnknownFailRequest}

// Policies a classifier or decision condition sets with on_error.
const (
	RuleOnErrorNoMatch = "no_match"
	RuleOnErrorMatch   = "match"
)

func (p UnknownPolicy) IsValid() bool {
	return slices.Contains(UnknownPolicies, p)
}

func UnknownPolicyChoices() string {
	names := make([]string, len(UnknownPolicies))
	for i, policy := range UnknownPolicies {
		names[i] = string(policy)
	}
	return strings.Join(names[:len(names)-1], ", ") + ", or " + names[len(names)-1]
}

const DecisionActionRoute = "route"

// DecisionAction is an explicit action a matched decision applies instead of
// candidate ranking. The only supported type is "route": send the request to
// Destination, overriding a caller-pinned model, so a detected prompt attack
// cannot bypass the guard by naming a model. Destination must resolve in
// model_config and the decision's rules must reference a jailbreak signal.
type DecisionAction struct {
	Type        string `yaml:"type" json:"type" jsonschema:"required"`
	Destination string `yaml:"destination" json:"destination" jsonschema:"required"`
}

// Decision represents a routing decision that combines multiple rules with boolean logic.
type Decision struct {
	Name               string                     `yaml:"name"`
	Description        string                     `yaml:"description,omitempty"`
	Priority           int                        `yaml:"priority,omitempty"`
	Tier               int                        `yaml:"tier,omitempty"`
	OutputContract     string                     `yaml:"output_contract,omitempty" json:"output_contract,omitempty"`
	OutputContractSpec *OutputContractSpec        `yaml:"output_contract_spec,omitempty" json:"output_contract_spec,omitempty"`
	Rules              RuleCombination            `yaml:"rules"`
	Action             *DecisionAction            `yaml:"action,omitempty" json:"action,omitempty"`
	ModelRefs          []ModelRef                 `yaml:"modelRefs,omitempty"`
	Algorithm          *AlgorithmConfig           `yaml:"algorithm,omitempty"`
	Adaptations        DecisionAdaptationsConfig  `yaml:"adaptations,omitempty"`
	Reliability        *DecisionReliability       `yaml:"reliability,omitempty" json:"reliability,omitempty"`
	Fallback           *fallback.FallbackOverride `yaml:"fallback,omitempty" json:"fallback,omitempty"`
	// Plugins are the decision's effective plugins: its own layered over its
	// recipe's and global.plugins (DeclaredPlugins are its own).
	Plugins             []DecisionPlugin           `yaml:"plugins,omitempty"`
	CandidateIterations []CandidateIterationConfig `yaml:"candidateIterations,omitempty"`
	// Retention says what the Router keeps of the decision's answers.
	Retention *DecisionRetention `yaml:"retention,omitempty" json:"retention,omitempty"`
	// Output says which media the decision's answers carry; OutputContract
	// and OutputContractSpec shape the text of an answer.
	Output *DecisionOutput `yaml:"output,omitempty" json:"output,omitempty"`
	// Annotations carries bounded, non-executable decision metadata for replay
	// and transport projections. Executable behavior belongs in Plugins or the
	// decision's attributes.
	Annotations map[string]interface{} `yaml:"annotations,omitempty" json:"annotations,omitempty"`

	// declaredPlugins are the decision's own plugins once the loader layered
	// Plugins (pluginsLayered).
	declaredPlugins []DecisionPlugin
	pluginsLayered  bool
}

// DecisionRetention expresses keep / drop / prefer-retain semantics over the
// Router-owned response content of a decision. All fields are tri-state
// pointers so we can distinguish "unset" from an explicit zero value.
//
// Runtime consumes Drop (response-cache, memory, and Responses-object write
// suppression), TTLTurns (per-entry cache TTL override), and KeepCurrentModel
// (model-switch-gate forced stay). Drop does not delete existing objects or
// disable explicit history reads, telemetry, or backend-side persistence.
// PreferPrefixRetention is only reported; its session-aware scoring bias and
// KV-cache eviction integration are follow-up work. All set fields are also
// observed via log + trace attributes and reported in the retention member of
// x-vsr-trace (issues #1747, #2009).
type DecisionRetention struct {
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

// AlgorithmConfig is how a decision uses its candidate models: a selector
// chooses one before calling it, an executor calls several and returns one
// response. Each type's settings are the block named after it; Budget and
// OnError are the executors' shared settings.
type AlgorithmConfig struct {
	Type              string                      `yaml:"type"`
	MinimumCandidates int                         `yaml:"minimum_candidates,omitempty"`
	Budget            *AlgorithmBudget            `yaml:"budget,omitempty"`
	OnError           string                      `yaml:"on_error,omitempty"`
	MultiFactor       *MultiFactorSelectionConfig `yaml:"multi_factor,omitempty"`
	SystemOne         *SystemOneSelectionConfig   `yaml:"systemone,omitempty"`
	Escalate          *EscalateAlgorithmConfig    `yaml:"escalate,omitempty"`
	Fusion            *FusionAlgorithmConfig      `yaml:"fusion,omitempty"`
	ReMoM             *ReMoMAlgorithmConfig       `yaml:"remom,omitempty"`
	Ratings           *RatingsAlgorithmConfig     `yaml:"ratings,omitempty"`
	Workflows         *WorkflowsAlgorithmConfig   `yaml:"workflows,omitempty"`
	// Extensions holds the blocks of algorithm types registered outside the
	// Router, each under its type's name.
	Extensions map[string]*StructuredPayload `yaml:",inline" json:"-" jsonschema:"-"`
}

// extensionFields are the keys Extensions may hold: the registered
// algorithm types the Router does not build in.
func (*AlgorithmConfig) extensionFields() []string { return extensionAlgorithmTypes() }

type HybridWeightsConfig struct {
	LogprobWeight float64 `yaml:"logprob_weight,omitempty"`
	MarginWeight  float64 `yaml:"margin_weight,omitempty"`
}

type RatingsAlgorithmConfig struct {
	MaxConcurrent int `yaml:"max_concurrent,omitempty"`
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
	MaxCompletionTokens          *int    `yaml:"max_completion_tokens,omitempty"`
	RoundTimeoutSeconds          int     `yaml:"round_timeout_seconds,omitempty"`
	MinSuccessfulResponses       int     `yaml:"min_successful_responses,omitempty"`
	ShuffleSeed                  int     `yaml:"shuffle_seed,omitempty"`
	IncludeIntermediateResponses bool    `yaml:"include_intermediate_responses,omitempty"`
	MaxResponsesPerRound         int     `yaml:"max_responses_per_round,omitempty"`
}

type ModelReasoningControl struct {
	UseReasoning         *bool  `yaml:"use_reasoning,omitempty"`
	ReasoningDescription string `yaml:"reasoning_description,omitempty"`
	ReasoningMode        string `yaml:"reasoning_mode,omitempty"`
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
	Type       string            `yaml:"type,omitempty" json:"type,omitempty"`
	Name       string            `yaml:"name,omitempty" json:"name,omitempty"`
	Label      string            `yaml:"label,omitempty" json:"label,omitempty"`
	Predicate  *NumericPredicate `yaml:"predicate,omitempty" json:"predicate,omitempty"`
	OnError    string            `yaml:"on_error,omitempty" json:"on_error,omitempty"`
	OnUnknown  UnknownPolicy     `yaml:"on_unknown,omitempty" json:"on_unknown,omitempty"`
	Operator   string            `yaml:"operator,omitempty" json:"operator,omitempty"`
	Conditions []RuleNode        `yaml:"conditions,omitempty" json:"conditions,omitempty"`
}

func (n *RuleNode) IsLeaf() bool {
	return n.Type != ""
}

// IsEmpty reports whether a rule node was omitted entirely. At a decision
// root, this is the canonical YAML representation of an unconditional
// terminal decision. Evaluators must not infer that meaning for nested nodes.
func (n *RuleNode) IsEmpty() bool {
	return n.Type == "" && n.Name == "" && n.Operator == "" && len(n.Conditions) == 0
}

// IsCatchAll reports a decision that matches every request: omitted rules or
// an explicit AND with no conditions.
func (n *RuleNode) IsCatchAll() bool {
	if n.IsEmpty() {
		return true
	}
	return !n.IsLeaf() && strings.EqualFold(n.Operator, RuleOperatorAnd) && len(n.Conditions) == 0
}

type (
	RuleCombination = RuleNode
	RuleCondition   = RuleNode
)
