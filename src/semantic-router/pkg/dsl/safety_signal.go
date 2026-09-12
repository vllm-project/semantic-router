package dsl

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"gopkg.in/yaml.v2"
)

func (c *Compiler) compileSafetySignal(s *SignalDecl) {
	payload := fieldsToMap(s.Fields)
	payload["name"] = s.Name
	raw, err := yaml.Marshal(payload)
	if err != nil {
		c.addError(s.Pos, "failed to encode safety signal %q: %v", s.Name, err)
		return
	}
	var rule config.SafetyRule
	if err := yaml.UnmarshalStrict(raw, &rule); err != nil {
		c.addError(s.Pos, "failed to decode safety signal %q: %v", s.Name, err)
		return
	}
	c.config.SafetyRules = append(c.config.SafetyRules, rule)
}

func (d *decompiler) safetyToSignal(rule *config.SafetyRule) *SignalDecl {
	fields := map[string]Value{
		"model":     StringValue{V: rule.Model},
		"threshold": FloatValue{V: rule.Threshold},
	}
	if rule.Description != "" {
		fields["description"] = StringValue{V: rule.Description}
	}
	if len(rule.Labels) > 0 {
		fields["labels"] = stringsToArray(rule.Labels)
	}
	if len(rule.UnsafeLabels) > 0 {
		fields["unsafe_labels"] = stringsToArray(rule.UnsafeLabels)
	}
	if h := rule.Hazard; h != nil {
		fields["hazard"] = ObjectValue{Fields: map[string]Value{
			"model": StringValue{V: h.Model}, "labels": stringsToArray(h.Labels),
			"categories": stringsToArray(h.Categories), "threshold": FloatValue{V: h.Threshold},
		}}
	}
	return &SignalDecl{SignalType: config.SignalTypeSafety, Name: rule.Name, Fields: fields}
}

func (d *decompiler) decompileSafetySignals() {
	for i := range d.cfg.SafetyRules {
		signal := d.safetyToSignal(&d.cfg.SafetyRules[i])
		d.write("SIGNAL safety %s {\n", quoteName(signal.Name))
		// Fixed order keeps canonical DSL output stable.
		for _, key := range []string{"description", "model", "labels", "unsafe_labels", "threshold", "hazard"} {
			if value, ok := signal.Fields[key]; ok {
				d.write("  %s: %s\n", key, formatPluginConfigValue(valueToInterface(value)))
			}
		}
		d.write("}\n\n")
	}
}
