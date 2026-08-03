package k8s

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestValidateDecisionDiagnosticsPluginConfiguration(t *testing.T) {
	require.NoError(t, validatePluginConfiguration("decision_diagnostics", []byte(`{
		"enabled": true,
		"max_signals": 32,
		"max_projections": 16,
		"max_text_runes": 128,
		"max_payload_bytes": 16384
	}`)))

	err := validatePluginConfiguration("decision_diagnostics", []byte(`{
		"enabled": true,
		"max_signals": 129
	}`))
	require.ErrorContains(t, err, "max_signals")

	err = validatePluginConfiguration("decision_diagnostics", []byte(`{
		"enabled": true,
		"request_body": "must not be accepted"
	}`))
	require.Error(t, err)
}
