package config

const (
	DecisionPluginMasking = "masking"

	// MaskingPlaceholderIndexToken is substituted with the per-request,
	// per-entity-type occurrence index. A template without it would collapse
	// distinct values onto one placeholder, so the validator requires it.
	MaskingPlaceholderIndexToken = "{index}"
)

// MaskingPluginConfig replaces detected PII in the provider-bound request with
// placeholders. See issue #3566. Failure is always closed (D4): there is
// deliberately no fail-open option for a privacy control.
type MaskingPluginConfig struct {
	Enabled bool `json:"enabled" yaml:"enabled"`
	// Threshold is the minimum classifier confidence for a span to be masked.
	// Zero means fall back to the router's configured PII model threshold.
	Threshold float32 `json:"threshold,omitempty" yaml:"threshold,omitempty"`
	// EntityTypes is an INCLUDE list: only these types are masked. Empty means
	// mask every detected type. Note this is the opposite of PIIRule's
	// pii_types_allowed, which is an allow-list of types permitted through.
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
