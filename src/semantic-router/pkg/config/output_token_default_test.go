package config

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v2"
)

func TestOutputTokenDefaultScalarContract(t *testing.T) {
	for _, wire := range []string{`"auto"`, `4096`} {
		var value OutputTokenDefault
		require.NoError(t, json.Unmarshal([]byte(wire), &value))
		out, err := json.Marshal(value)
		require.NoError(t, err)
		require.Equal(t, wire, string(out))
		encoded, err := yaml.Marshal(value)
		require.NoError(t, err)
		var again OutputTokenDefault
		require.NoError(t, yaml.Unmarshal(encoded, &again))
		require.Equal(t, value, again)
	}
	for _, wire := range []string{`0`, `-1`, `1.5`, `"AUTO"`, `true`, `{}`, `null`} {
		var value OutputTokenDefault
		require.Error(t, json.Unmarshal([]byte(wire), &value), wire)
	}
}

func TestAutomaticOutputPolicyCanonicalRoundTrip(t *testing.T) {
	body := []byte(`routing:
  decisions:
    - name: capacity
      algorithm:
        type: multi_factor
        multi_factor:
          expected_output_tokens: 4096
      plugins:
        - type: request_params
          configuration:
            default_max_tokens: auto
`)
	cfg, err := ParseRoutingYAMLBytes(body)
	require.NoError(t, err)
	encoded, err := yaml.Marshal(CanonicalConfigFromRouterConfig(cfg))
	require.NoError(t, err)
	again, err := ParseYAMLBytes(encoded)
	require.NoError(t, err)
	require.True(t, again.Decisions[0].GetRequestParamsConfig().DefaultMaxTokens.IsAuto())
	require.Equal(t, 4096, *again.Decisions[0].Algorithm.MultiFactor.ExpectedOutputTokens)
}
