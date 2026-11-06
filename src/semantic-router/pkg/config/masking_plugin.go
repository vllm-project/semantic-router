package config

import (
	"fmt"
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

const (
	DecisionPluginMasking = "masking"

	// MaskingPlaceholderIndexToken is substituted with the per-request,
	// per-entity-type occurrence index. A template without it would collapse
	// distinct values onto one placeholder, so the validator requires it.
	MaskingPlaceholderIndexToken = "{index}"
)

// MaskingPluginConfig replaces personal data in the provider-bound request
// with placeholders. See issue #3566. Failure is always closed (D4): there is
// deliberately no fail-open option for a privacy control.
type MaskingPluginConfig struct {
	Enabled bool `json:"enabled" yaml:"enabled"`
	// Signal names the pii rule of the decision's routing scope that finds
	// the personal data: its detector, its threshold, and its
	// pii_types_allowed, which are never masked.
	Signal string `json:"signal" yaml:"signal"`
	// EntityTypes is an INCLUDE list: only these types are masked. Empty means
	// mask every type the rule counts.
	EntityTypes []string `json:"entity_types,omitempty" yaml:"entity_types,omitempty"`
	// Placeholders maps an entity type to a template containing {index}.
	// Types absent from the map use MaskingDefaultPlaceholder.
	Placeholders map[string]string `json:"placeholders,omitempty" yaml:"placeholders,omitempty"`
}

// GetMaskingConfig returns the masking plugin configuration, or nil when the
// decision does not configure it.
func (d *Decision) GetMaskingConfig() *MaskingPluginConfig {
	result := &MaskingPluginConfig{}
	return decodeDecisionPlugin(d, DecisionPluginMasking, result)
}

// EffectivePlaceholder returns the template for an entity type, defaulting to
// the same "[TYPE_n]" shape the classification API already emits.
func (c *MaskingPluginConfig) EffectivePlaceholder(entityType string) string {
	if c != nil {
		if template, ok := c.Placeholders[entityType]; ok && template != "" {
			return template
		}
	}
	return "[" + entityType + "_" + MaskingPlaceholderIndexToken + "]"
}

// MaskingEntityTypes are the labels the pii preset produces, which are the
// entity types masking can name.
func MaskingEntityTypes() []string {
	preset, _ := presets.Lookup(presets.PII, presets.StageRequest)
	return preset.OptionKeys()
}

func validateMaskingPlugin(at PluginAt, p *MaskingPluginConfig) error {
	if strings.TrimSpace(p.Signal) == "" {
		return at.Errorf("signal is required: name the pii rule whose detections are masked")
	}
	labels := MaskingEntityTypes()
	for _, entityType := range p.EntityTypes {
		if !slices.Contains(labels, entityType) {
			return at.Errorf("entity_types: %q is not a label the pii preset produces (%s)", entityType, strings.Join(labels, ", "))
		}
	}
	for entityType, template := range p.Placeholders {
		if !slices.Contains(labels, entityType) {
			return at.Errorf("placeholders: %q is not a label the pii preset produces (%s)", entityType, strings.Join(labels, ", "))
		}
		// Without {index} two distinct values collapse onto one placeholder,
		// which the issue's determinism criterion forbids (#3566).
		if !strings.Contains(template, MaskingPlaceholderIndexToken) {
			return at.Errorf("placeholder for %q must contain %s", entityType, MaskingPlaceholderIndexToken)
		}
	}
	return nil
}

// validateMaskingSignals resolves every masking plugin of a routing scope to
// the pii rule it names.
func validateMaskingSignals(cfg *RouterConfig) error {
	for i := range cfg.Decisions {
		masking := cfg.Decisions[i].GetMaskingConfig()
		if masking == nil || cfg.PIIRule(masking.Signal) != nil {
			continue
		}
		return fmt.Errorf("decision %q masking: signal %q names no pii rule of this recipe", cfg.Decisions[i].Name, masking.Signal)
	}
	return nil
}

// PIIRule returns the pii rule of this routing scope named name.
func (c *RouterConfig) PIIRule(name string) *PIIRule {
	for i := range c.PIIRules {
		if c.PIIRules[i].Name == name {
			return &c.PIIRules[i]
		}
	}
	return nil
}

// MaskingSignals are the pii rules the enabled masking plugins of this
// routing scope name, in decision order.
func (c *RouterConfig) MaskingSignals() []string {
	var names []string
	for i := range c.Decisions {
		masking := c.Decisions[i].GetMaskingConfig()
		if masking != nil && masking.Enabled && masking.Signal != "" && !slices.Contains(names, masking.Signal) {
			names = append(names, masking.Signal)
		}
	}
	return names
}
