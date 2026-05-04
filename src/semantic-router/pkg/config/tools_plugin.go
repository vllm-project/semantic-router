package config

const (
	DecisionPluginTools = "tools"

	ToolsPluginModeNone        = "none"
	ToolsPluginModePassthrough = "passthrough"
	ToolsPluginModeFiltered    = "filtered"

	// ToolsStrategyDefault is the strategy name used when no explicit strategy
	// is configured.  It resolves to the embedding-similarity retriever.
	ToolsStrategyDefault = "default"

	// Dynamic retrieval strategy names.  These are decision-scoped knobs that
	// describe how the retriever combines semantic similarity with optional
	// per-decision tool-call history.  Behavior is implemented in subsequent
	// issues; this contract only fixes the configuration surface.
	DynamicRetrievalStrategySemanticOnly  = "semantic_only"
	DynamicRetrievalStrategyHybridHistory = "hybrid_history"
)

// ToolsPluginConfig represents per-decision tool handling.
// It owns both request tool filtering and semantic tool selection behavior.
type ToolsPluginConfig struct {
	Enabled           bool     `json:"enabled" yaml:"enabled"`
	Mode              string   `json:"mode,omitempty" yaml:"mode,omitempty"`
	SemanticSelection *bool    `json:"semantic_selection,omitempty" yaml:"semantic_selection,omitempty"`
	AllowTools        []string `json:"allow_tools,omitempty" yaml:"allow_tools,omitempty"`
	BlockTools        []string `json:"block_tools,omitempty" yaml:"block_tools,omitempty"`
	// Strategy names the retriever strategy to use for semantic tool selection.
	// When empty, EffectiveStrategy returns ToolsStrategyDefault.
	// The value must match a name registered in the router's tools.Registry.
	Strategy string `json:"strategy,omitempty" yaml:"strategy,omitempty"`
	// DynamicRetrieval configures history-aware tool retrieval behavior.  When
	// nil or with Enabled=false, the existing semantic-similarity retriever is
	// used unchanged.  See issue #1832 for the contract scope; implementation
	// of the strategies and the persisted-priors layer is staged in #1839.
	DynamicRetrieval *DynamicRetrievalConfig `json:"dynamic_retrieval,omitempty" yaml:"dynamic_retrieval,omitempty"`
}

// DynamicRetrievalConfig is the decision-scoped contract for history-aware
// tool retrieval.  It is intentionally additive: when nil, when Enabled is
// false, or when omitted from configuration, the router behaves exactly as
// before this contract was introduced.
type DynamicRetrievalConfig struct {
	// Enabled gates the entire dynamic retrieval path.  When false, none of
	// the other fields are consulted and the validator skips range checks.
	Enabled bool `json:"enabled" yaml:"enabled"`
	// Strategy selects the retrieval algorithm.  Must be one of
	// DynamicRetrievalStrategySemanticOnly or DynamicRetrievalStrategyHybridHistory.
	// When empty, EffectiveStrategy returns DynamicRetrievalStrategySemanticOnly.
	Strategy string `json:"strategy,omitempty" yaml:"strategy,omitempty"`
	// HistoryWindow is the number of recent tool-call observations the
	// retriever may inspect when Strategy is hybrid_history.  Required to be
	// at least 1 when the hybrid strategy is active.  Ignored otherwise.
	HistoryWindow int `json:"history_window,omitempty" yaml:"history_window,omitempty"`
	// Weights blends the per-source signals when Strategy is hybrid_history.
	// When nil, sensible defaults are used by the retriever (semantic 1.0,
	// history 1.0, decision_prior 0.0, repetition_penalty 0.0).
	Weights *DynamicRetrievalWeights `json:"weights,omitempty" yaml:"weights,omitempty"`
	// MinHistoryConfidence is the per-decision confidence threshold below
	// which the retriever treats history evidence as too weak to use.  Must
	// be in [0.0, 1.0].
	MinHistoryConfidence float64 `json:"min_history_confidence,omitempty" yaml:"min_history_confidence,omitempty"`
	// FallbackOnLowConfidence, when true, instructs the retriever to fall
	// back to the semantic-only ranking whenever the history evidence does
	// not clear MinHistoryConfidence.
	FallbackOnLowConfidence bool `json:"fallback_on_low_confidence,omitempty" yaml:"fallback_on_low_confidence,omitempty"`
}

// DynamicRetrievalWeights holds the combination weights for the hybrid
// retriever.  Each weight is a non-negative scalar; the retriever normalizes
// internally before applying them.  Zero weights mean the corresponding
// signal is ignored.
type DynamicRetrievalWeights struct {
	Semantic          float64 `json:"semantic,omitempty" yaml:"semantic,omitempty"`
	History           float64 `json:"history,omitempty" yaml:"history,omitempty"`
	DecisionPrior     float64 `json:"decision_prior,omitempty" yaml:"decision_prior,omitempty"`
	RepetitionPenalty float64 `json:"repetition_penalty,omitempty" yaml:"repetition_penalty,omitempty"`
}

// GetToolsConfig returns the tools plugin configuration.
func (d *Decision) GetToolsConfig() *ToolsPluginConfig {
	result := &ToolsPluginConfig{}
	return decodeDecisionPlugin(d, DecisionPluginTools, result)
}

// EffectiveMode returns the defaulted tool-handling mode.
func (c *ToolsPluginConfig) EffectiveMode() string {
	if c == nil || c.Mode == "" {
		return ToolsPluginModePassthrough
	}
	return c.Mode
}

// EffectiveStrategy returns the retriever strategy name, defaulting to
// ToolsStrategyDefault when none is configured.
func (c *ToolsPluginConfig) EffectiveStrategy() string {
	if c == nil || c.Strategy == "" {
		return ToolsStrategyDefault
	}
	return c.Strategy
}

// SelectionEnabled reports whether semantic tool selection is allowed for this plugin.
func (c *ToolsPluginConfig) SelectionEnabled() bool {
	if c == nil || !c.Enabled || c.EffectiveMode() == ToolsPluginModeNone {
		return false
	}
	if c.SemanticSelection == nil {
		return true
	}
	return *c.SemanticSelection
}

// DynamicRetrievalEnabled reports whether the decision opts in to dynamic
// retrieval.  Returns false for nil receivers and for configurations where
// the dynamic_retrieval block is unset or disabled.
func (c *ToolsPluginConfig) DynamicRetrievalEnabled() bool {
	if c == nil || c.DynamicRetrieval == nil {
		return false
	}
	return c.DynamicRetrieval.Enabled
}

// EffectiveStrategy returns the dynamic retrieval strategy, defaulting to
// DynamicRetrievalStrategySemanticOnly when the receiver is nil or has no
// strategy configured.  The default mirrors the pre-#1832 behavior so that
// callers can rely on a non-empty strategy name without nil-checking.
func (d *DynamicRetrievalConfig) EffectiveStrategy() string {
	if d == nil || d.Strategy == "" {
		return DynamicRetrievalStrategySemanticOnly
	}
	return d.Strategy
}
