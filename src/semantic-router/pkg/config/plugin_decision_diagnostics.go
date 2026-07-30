package config

import "fmt"

const (
	defaultDecisionDiagnosticsMaxSignals      = 32
	defaultDecisionDiagnosticsMaxProjections  = 16
	defaultDecisionDiagnosticsMaxTextRunes    = 128
	defaultDecisionDiagnosticsMaxPayloadBytes = 16 * 1024
	maximumDecisionDiagnosticsMaxSignals      = 128
	maximumDecisionDiagnosticsMaxProjections  = 64
	maximumDecisionDiagnosticsMaxTextRunes    = 512
	maximumDecisionDiagnosticsMaxPayloadBytes = 64 * 1024
)

// DecisionDiagnosticsPluginConfig controls the bounded, content-free routing
// diagnostics emitted as ExtProc dynamic metadata for following filters.
type DecisionDiagnosticsPluginConfig struct {
	Enabled         bool `json:"enabled" yaml:"enabled"`
	MaxSignals      int  `json:"max_signals,omitempty" yaml:"max_signals,omitempty"`
	MaxProjections  int  `json:"max_projections,omitempty" yaml:"max_projections,omitempty"`
	MaxTextRunes    int  `json:"max_text_runes,omitempty" yaml:"max_text_runes,omitempty"`
	MaxPayloadBytes int  `json:"max_payload_bytes,omitempty" yaml:"max_payload_bytes,omitempty"`
}

// GetDecisionDiagnosticsConfig returns the enabled plugin configuration with
// safe defaults applied. A missing or disabled plugin returns nil.
func (d *Decision) GetDecisionDiagnosticsConfig() *DecisionDiagnosticsPluginConfig {
	result := &DecisionDiagnosticsPluginConfig{}
	result = decodeDecisionPlugin(d, DecisionPluginDecisionDiagnostics, result)
	if result == nil || !result.Enabled {
		return nil
	}
	result.applyDefaults()
	return result
}

func (c *DecisionDiagnosticsPluginConfig) applyDefaults() {
	if c.MaxSignals == 0 {
		c.MaxSignals = defaultDecisionDiagnosticsMaxSignals
	}
	if c.MaxProjections == 0 {
		c.MaxProjections = defaultDecisionDiagnosticsMaxProjections
	}
	if c.MaxTextRunes == 0 {
		c.MaxTextRunes = defaultDecisionDiagnosticsMaxTextRunes
	}
	if c.MaxPayloadBytes == 0 {
		c.MaxPayloadBytes = defaultDecisionDiagnosticsMaxPayloadBytes
	}
}

// Validate applies defaults and rejects values above the hard safety ceiling.
func (c *DecisionDiagnosticsPluginConfig) Validate() error {
	if c == nil {
		return nil
	}
	c.applyDefaults()
	if err := validateDecisionDiagnosticsBound("max_signals", c.MaxSignals, maximumDecisionDiagnosticsMaxSignals); err != nil {
		return err
	}
	if err := validateDecisionDiagnosticsBound("max_projections", c.MaxProjections, maximumDecisionDiagnosticsMaxProjections); err != nil {
		return err
	}
	if err := validateDecisionDiagnosticsBound("max_text_runes", c.MaxTextRunes, maximumDecisionDiagnosticsMaxTextRunes); err != nil {
		return err
	}
	return validateDecisionDiagnosticsBound("max_payload_bytes", c.MaxPayloadBytes, maximumDecisionDiagnosticsMaxPayloadBytes)
}

func validateDecisionDiagnosticsPlugin(decision *Decision) error {
	if decision == nil {
		return nil
	}
	plugin := decision.GetPlugin(DecisionPluginDecisionDiagnostics)
	if plugin == nil {
		return nil
	}
	if plugin.Configuration == nil {
		return fmt.Errorf("decision_diagnostics plugin configuration is required")
	}

	cfg := &DecisionDiagnosticsPluginConfig{}
	if err := UnmarshalPluginConfig(plugin.Configuration, cfg); err != nil {
		return fmt.Errorf("invalid decision_diagnostics plugin configuration: %w", err)
	}
	return cfg.Validate()
}

func validateDecisionDiagnosticsBound(name string, value, maximum int) error {
	if value < 1 || value > maximum {
		return fmt.Errorf("decision_diagnostics %s must be between 1 and %d", name, maximum)
	}
	return nil
}
