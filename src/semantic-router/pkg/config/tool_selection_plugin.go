package config

const (
	ToolSelectionModeAdd    = "add"
	ToolSelectionModeFilter = "filter"
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
