package dsl

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

func (v *Validator) checkPromptCachePluginRefConstraints(plugin *PluginRef) {
	for _, template := range v.prog.Plugins {
		if template.Name != plugin.Name {
			continue
		}
		if config.NormalizeDecisionPluginType(template.PluginType) != config.DecisionPluginPromptCache {
			return
		}
		if len(plugin.Fields) == 0 {
			return
		}
		fields := make(map[string]Value, len(template.Fields)+len(plugin.Fields))
		for name, value := range template.Fields {
			fields[name] = value
		}
		for name, value := range plugin.Fields {
			fields[name] = value
		}
		v.checkPromptCachePluginConstraints(plugin.Pos, fields)
		return
	}
	if config.NormalizeDecisionPluginType(plugin.Name) == config.DecisionPluginPromptCache {
		v.checkPromptCachePluginConstraints(plugin.Pos, plugin.Fields)
	}
}

func (v *Validator) checkPromptCachePluginConstraints(
	position Position,
	fields map[string]Value,
) {
	if _, err := decodePromptCachePluginFields(fields); err != nil {
		v.addDiag(DiagConstraint, position, err.Error(), nil)
	}
}
