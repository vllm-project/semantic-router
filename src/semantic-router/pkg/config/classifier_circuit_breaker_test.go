package config

import (
	"testing"

	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v2"
)

func TestRemoteClassifierCircuitBreakerValidation(t *testing.T) {
	for _, value := range []int{0, -1} {
		for _, cb := range []*ClassifierCircuitBreaker{{Enabled: true, ConsecutiveFailures: &value}, {Enabled: true, OpenIntervalMs: &value}} {
			backend := &RemoteClassifierBackend{Protocol: RemoteClassifierProtocolHTTPClassify, Model: "remote", CircuitBreaker: cb}
			require.Error(t, backend.Validate())
			guard := PromptGuardConfig{Backend: backend}
			require.Error(t, validatePromptGuardBackendConfig(&guard))
			cfg := &RouterConfig{ExternalModels: []ExternalModelConfig{{Name: "remote", CircuitBreaker: cb}}}
			require.Error(t, validateExternalClassifierCircuitBreakers(cfg))
		}
	}
	var backend RemoteClassifierBackend
	require.NoError(t, yaml.Unmarshal([]byte("protocol: http_classify\nmodel: remote\ncircuit_breaker:\n  enabled: true\n"), &backend))
	require.NoError(t, backend.Validate())
	n, ms := backend.CircuitBreaker.EffectiveValues()
	require.Equal(t, 3, n)
	require.Equal(t, 30000, ms)
	require.Nil(t, backend.CircuitBreaker.ConsecutiveFailures, "defaults mutated canonical input")
	backend.CircuitBreaker.Enabled = false
	n, ms = backend.CircuitBreaker.EffectiveValues()
	require.Zero(t, n)
	require.Zero(t, ms)
}
