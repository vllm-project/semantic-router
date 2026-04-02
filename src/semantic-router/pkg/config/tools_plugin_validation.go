package config

import "fmt"

func (c *ToolsPluginConfig) Validate() error {
	if c == nil || !c.Enabled {
		return nil
	}

	mode := c.EffectiveMode()
	switch mode {
	case ToolsPluginModeNone, ToolsPluginModePassthrough, ToolsPluginModeFiltered:
	default:
		return fmt.Errorf("tools plugin: mode must be one of %q, %q, or %q", ToolsPluginModeNone, ToolsPluginModePassthrough, ToolsPluginModeFiltered)
	}

	hasLists := len(c.AllowTools) > 0 || len(c.BlockTools) > 0
	switch mode {
	case ToolsPluginModeFiltered:
		if !hasLists {
			return fmt.Errorf("tools plugin: mode=%q requires allow_tools or block_tools", ToolsPluginModeFiltered)
		}
	default:
		if hasLists {
			return fmt.Errorf("tools plugin: allow_tools/block_tools require mode=%q", ToolsPluginModeFiltered)
		}
	}

	return nil
}
