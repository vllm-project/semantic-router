package config

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestEffectiveDecisionDiagnosticsConfigAppliesBoundedDefaults(t *testing.T) {
	decision := Decision{Plugins: []DecisionPlugin{{
		Type: DecisionPluginDecisionDiagnostics,
		Configuration: MustStructuredPayload(map[string]interface{}{
			"enabled": true,
		}),
	}}}

	got := decision.GetDecisionDiagnosticsConfig()
	require.NotNil(t, got)
	require.True(t, got.Enabled)
	require.Equal(t, 32, got.MaxSignals)
	require.Equal(t, 16, got.MaxProjections)
	require.Equal(t, 128, got.MaxTextRunes)
	require.Equal(t, 16*1024, got.MaxPayloadBytes)
}

func TestEffectiveDecisionDiagnosticsConfigReturnsNilWhenDisabled(t *testing.T) {
	decision := Decision{Plugins: []DecisionPlugin{{
		Type: DecisionPluginDecisionDiagnostics,
		Configuration: MustStructuredPayload(map[string]interface{}{
			"enabled": false,
		}),
	}}}

	require.Nil(t, decision.GetDecisionDiagnosticsConfig())
}

func TestDecisionDiagnosticsConfigRejectsUnsafeBounds(t *testing.T) {
	tests := []struct {
		name   string
		config map[string]interface{}
	}{
		{name: "too many signals", config: map[string]interface{}{"enabled": true, "max_signals": 129}},
		{name: "too many projections", config: map[string]interface{}{"enabled": true, "max_projections": 65}},
		{name: "text too long", config: map[string]interface{}{"enabled": true, "max_text_runes": 513}},
		{name: "payload too large", config: map[string]interface{}{"enabled": true, "max_payload_bytes": 65537}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			decision := Decision{Name: "route", Plugins: []DecisionPlugin{{
				Type:          DecisionPluginDecisionDiagnostics,
				Configuration: MustStructuredPayload(tt.config),
			}}}
			err := validateDecisionDiagnosticsPlugin(&decision)
			require.Error(t, err)
		})
	}
}

func TestDecisionDiagnosticsIsPartOfSupportedPluginSurface(t *testing.T) {
	require.True(t, IsSupportedDecisionPluginType(DecisionPluginDecisionDiagnostics))
}
