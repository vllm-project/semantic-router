package dsl

import (
	"fmt"

	"gopkg.in/yaml.v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// applicationSignal reads and writes the SIGNAL blocks of one application
// signal type. A block's fields are its rule's YAML fields: it decodes the
// way the rule decodes from YAML, and a field the rule does not have is an
// error.
type applicationSignal struct {
	compile func(c *Compiler, s *SignalDecl)
	rules   func(cfg *config.RouterConfig) []interface{}
}

var applicationSignals = map[string]applicationSignal{
	config.SignalTypeDomain: applicationSignalOf(config.SignalTypeDomain,
		func(cfg *config.RouterConfig) *[]config.Category { return &cfg.Categories }),
	config.SignalTypeFactCheck: applicationSignalOf(config.SignalTypeFactCheck,
		func(cfg *config.RouterConfig) *[]config.FactCheckRule { return &cfg.FactCheckRules }),
	config.SignalTypeUserFeedback: applicationSignalOf(config.SignalTypeUserFeedback,
		func(cfg *config.RouterConfig) *[]config.UserFeedbackRule { return &cfg.UserFeedbackRules }),
	config.SignalTypeTurnRelation: applicationSignalOf(config.SignalTypeTurnRelation,
		func(cfg *config.RouterConfig) *[]config.TurnRelationRule { return &cfg.TurnRelationRules }),
	config.SignalTypeComplexity: applicationSignalOf(config.SignalTypeComplexity,
		func(cfg *config.RouterConfig) *[]config.ComplexityRule { return &cfg.ComplexityRules }),
	config.SignalTypeModality: applicationSignalOf(config.SignalTypeModality,
		func(cfg *config.RouterConfig) *[]config.ModalityRule { return &cfg.ModalityRules }),
	config.SignalTypeJailbreak: applicationSignalOf(config.SignalTypeJailbreak,
		func(cfg *config.RouterConfig) *[]config.JailbreakRule { return &cfg.JailbreakRules }),
	config.SignalTypeSafety: applicationSignalOf(config.SignalTypeSafety,
		func(cfg *config.RouterConfig) *[]config.SafetyRule { return &cfg.SafetyRules }),
	config.SignalTypePII: applicationSignalOf(config.SignalTypePII,
		func(cfg *config.RouterConfig) *[]config.PIIRule { return &cfg.PIIRules }),
	config.SignalTypeHallucination: applicationSignalOf(config.SignalTypeHallucination,
		func(cfg *config.RouterConfig) *[]config.HallucinationRule { return &cfg.HallucinationRules }),
	config.SignalTypeRefusal: applicationSignalOf(config.SignalTypeRefusal,
		func(cfg *config.RouterConfig) *[]config.RefusalRule { return &cfg.RefusalRules }),
}

func applicationSignalOf[T any](signalType string, collection func(*config.RouterConfig) *[]T) applicationSignal {
	return applicationSignal{
		compile: func(c *Compiler, s *SignalDecl) {
			payload := fieldsToMap(s.Fields)
			payload["name"] = s.Name
			raw, err := yaml.Marshal(payload)
			if err != nil {
				c.addError(s.Pos, "failed to encode %s signal %q: %v", signalType, s.Name, err)
				return
			}
			var rule T
			if err := yaml.UnmarshalStrict(raw, &rule); err != nil {
				c.addError(s.Pos, "%s signal %q: %v", signalType, s.Name, err)
				return
			}
			target := collection(c.config)
			*target = append(*target, rule)
		},
		rules: func(cfg *config.RouterConfig) []interface{} {
			declared := *collection(cfg)
			rules := make([]interface{}, len(declared))
			for index := range declared {
				rules[index] = declared[index]
			}
			return rules
		},
	}
}

// applicationRuleFields are the name of a rule and the fields its YAML form
// sets, in declaration order.
func applicationRuleFields(rule interface{}) (string, yaml.MapSlice) {
	raw, err := yaml.Marshal(rule)
	if err != nil {
		return "", nil
	}
	var decoded yaml.MapSlice
	if err := yaml.Unmarshal(raw, &decoded); err != nil {
		return "", nil
	}
	name := ""
	fields := make(yaml.MapSlice, 0, len(decoded))
	for _, field := range decoded {
		if field.Key == "name" {
			name = fmt.Sprint(field.Value)
			continue
		}
		if value := yamlFieldValue(field.Value); value != nil {
			fields = append(fields, yaml.MapItem{Key: field.Key, Value: value})
		}
	}
	return name, fields
}

// yamlFieldValue is a decoded YAML value as the DSL formatters take it,
// without the fields that are null.
func yamlFieldValue(raw interface{}) interface{} {
	switch typed := raw.(type) {
	case yaml.MapSlice:
		value := make(map[string]interface{}, len(typed))
		for _, item := range typed {
			if field := yamlFieldValue(item.Value); field != nil {
				value[fmt.Sprint(item.Key)] = field
			}
		}
		return value
	case []interface{}:
		items := make([]interface{}, len(typed))
		for index, item := range typed {
			items[index] = yamlFieldValue(item)
		}
		return items
	default:
		return typed
	}
}

func (d *decompiler) decompileApplicationSignals(signalType string) {
	for _, rule := range applicationSignals[signalType].rules(d.cfg) {
		name, fields := applicationRuleFields(rule)
		d.write("SIGNAL %s %s {\n", signalType, quoteName(name))
		for _, field := range fields {
			d.write("  %v: %s\n", field.Key, formatPluginConfigValue(field.Value))
		}
		d.write("}\n\n")
	}
}

func (d *decompiler) appendApplicationSignals(prog *Program, signalType string) {
	for _, rule := range applicationSignals[signalType].rules(d.cfg) {
		name, fields := applicationRuleFields(rule)
		decl := &SignalDecl{SignalType: signalType, Name: name, Fields: make(map[string]Value, len(fields))}
		for _, field := range fields {
			decl.Fields[fmt.Sprint(field.Key)] = interfaceValueToDSLValue(field.Value)
		}
		prog.Signals = append(prog.Signals, decl)
	}
}
