package config

const (
	ToolSelectionModeAdd    = "add"
	ToolSelectionModeFilter = "filter"

	// StickyToolSelectionDefaultMaxTools is the hard safety bound on the
	// number of tools a sticky session retains when sticky.max_tools is
	// unset. See issue #3347.
	StickyToolSelectionDefaultMaxTools = 16

	// StickyToolSelectionDefaultMaxNewToolsPerTurn bounds how many newly
	// relevant tools sticky selection may append in a single turn when
	// sticky.max_new_tools_per_turn is unset.
	StickyToolSelectionDefaultMaxNewToolsPerTurn = 2

	// StickyToolSelectionMaxToolsUpperBound is the hard ceiling accepted for
	// an explicit sticky.max_tools value.
	StickyToolSelectionMaxToolsUpperBound = 128
)

// ToolSelectionPluginConfig configures semantic tool add/filter on a matched decision.
// This is separate from the legacy "tools" plugin (passthrough/filtered/none routing).
type ToolSelectionPluginConfig struct {
	Enabled bool   `json:"enabled" yaml:"enabled"`
	Mode    string `json:"mode,omitempty" yaml:"mode,omitempty"`

	// --- Add mode (database retrieval) ---
	ToolsDBPath         string                       `json:"tools_db_path,omitempty" yaml:"tools_db_path,omitempty"`
	TopK                int                          `json:"top_k,omitempty" yaml:"top_k,omitempty"`
	SimilarityThreshold *float32                     `json:"similarity_threshold,omitempty" yaml:"similarity_threshold,omitempty"`
	AdvancedFiltering   *AdvancedToolFilteringConfig `json:"advanced_filtering,omitempty" yaml:"advanced_filtering,omitempty"`
	Strategy            string                       `json:"strategy,omitempty" yaml:"strategy,omitempty"`
	FallbackToEmpty     *bool                        `json:"fallback_to_empty,omitempty" yaml:"fallback_to_empty,omitempty"`

	// --- Filter mode (subset of request.tools) ---
	RelevanceThreshold *float32 `json:"relevance_threshold,omitempty" yaml:"relevance_threshold,omitempty"`
	PreserveCount      int      `json:"preserve_count,omitempty" yaml:"preserve_count,omitempty"`

	// --- Session-scoped sticky selection (opt-in, disabled by default) ---
	Sticky *StickyToolSelectionConfig `json:"sticky,omitempty" yaml:"sticky,omitempty"`
}

// StickyToolSelectionConfig configures opt-in, bounded session-scoped
// tool-set stickiness for the tool_selection plugin (issue #3347). Disabled
// by default: while Enabled is false, the plugin performs ordinary per-turn
// selection and never reads or writes session state.
//
// Only tool identities and bounded metadata are ever persisted under this
// config, in pkg/sessiontools — never full llmprotocol.Tool values,
// descriptions, JSON Schemas, arguments, results, prompts, authorization
// decisions, credentials, or raw session/principal identifiers.
type StickyToolSelectionConfig struct {
	Enabled bool `json:"enabled" yaml:"enabled"`

	// MaxTools and MaxNewToolsPerTurn are pointers, not plain ints like the
	// blueprint's original Go contract: for both, an explicit 0 is a
	// distinct, meaningful configured value from "unset" — max_tools: 0 must
	// be rejected by validation (valid range is 1..128), while
	// max_new_tools_per_turn: 0 is valid and means "no relevance-driven
	// growth, reuse and call-pinning only". A plain int's zero value cannot
	// carry either distinction from "the field was omitted". See PL-0042
	// task notes (tools/agent/docs/plans/pl-0042-sticky-tool-selection.md).
	MaxTools           *int  `json:"max_tools,omitempty" yaml:"max_tools,omitempty"`
	MaxNewToolsPerTurn *int  `json:"max_new_tools_per_turn,omitempty" yaml:"max_new_tools_per_turn,omitempty"`
	PinCalledTools     *bool `json:"pin_called_tools,omitempty" yaml:"pin_called_tools,omitempty"`
}

// EffectiveMaxTools returns the configured max_tools, or
// StickyToolSelectionDefaultMaxTools when unset.
func (s *StickyToolSelectionConfig) EffectiveMaxTools() int {
	if s == nil || s.MaxTools == nil {
		return StickyToolSelectionDefaultMaxTools
	}
	return *s.MaxTools
}

// EffectiveMaxNewToolsPerTurn returns the configured max_new_tools_per_turn,
// or StickyToolSelectionDefaultMaxNewToolsPerTurn when unset.
func (s *StickyToolSelectionConfig) EffectiveMaxNewToolsPerTurn() int {
	if s == nil || s.MaxNewToolsPerTurn == nil {
		return StickyToolSelectionDefaultMaxNewToolsPerTurn
	}
	return *s.MaxNewToolsPerTurn
}

// EffectivePinCalledTools returns the configured pin_called_tools, defaulting
// to true when unset.
func (s *StickyToolSelectionConfig) EffectivePinCalledTools() bool {
	if s == nil || s.PinCalledTools == nil {
		return true
	}
	return *s.PinCalledTools
}

func (d *Decision) GetToolSelectionConfig() *ToolSelectionPluginConfig {
	result := &ToolSelectionPluginConfig{}
	return decodeDecisionPlugin(d, DecisionPluginToolSelection, result)
}

func (c *ToolSelectionPluginConfig) EffectiveStrategy() string {
	if c == nil || c.Strategy == "" {
		return ToolsStrategyDefault
	}
	return c.Strategy
}
