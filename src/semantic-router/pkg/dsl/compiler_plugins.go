package dsl

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

func (c *Compiler) compilePluginRef(ref *PluginRef) *config.DecisionPlugin {
	// Check if it's a template reference
	if tmpl, ok := c.pluginTemplates[ref.Name]; ok {
		// Merge template fields with override fields
		mergedFields := make(map[string]Value)
		for k, v := range tmpl.Fields {
			mergedFields[k] = v
		}
		if ref.Fields != nil {
			for k, v := range ref.Fields {
				mergedFields[k] = v
			}
		}
		return c.buildDecisionPlugin(tmpl.PluginType, mergedFields)
	}

	// Inline plugin — ref.Name is the plugin type
	fields := ref.Fields
	if fields == nil {
		fields = make(map[string]Value)
	}
	return c.buildDecisionPlugin(ref.Name, fields)
}
