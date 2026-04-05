package config

const (
	DecisionPluginTools = "tools"

	ToolsPluginModeNone        = "none"
	ToolsPluginModePassthrough = "passthrough"
	ToolsPluginModeFiltered    = "filtered"
)

// ToolsPluginConfig represents per-decision tool handling.
// It owns both request tool filtering and semantic tool selection behavior.
type ToolsPluginConfig struct {
	Enabled           bool     `json:"enabled" yaml:"enabled"`
	Mode              string   `json:"mode,omitempty" yaml:"mode,omitempty"`
	SemanticSelection *bool    `json:"semantic_selection,omitempty" yaml:"semantic_selection,omitempty"`
	AllowTools        []string `json:"allow_tools,omitempty" yaml:"allow_tools,omitempty"`
	BlockTools        []string `json:"block_tools,omitempty" yaml:"block_tools,omitempty"`
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
