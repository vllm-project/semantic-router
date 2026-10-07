package k8s

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apis/vllm.ai/v1alpha1"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
)

func TestConvertDecisionCarriesReliabilityAndFallback(t *testing.T) {
	retries, enabled := int32(1), true
	decision, err := (&CRDConverter{}).convertDecision(v1alpha1.Decision{
		Name: "long_report",
		Reliability: &v1alpha1.DecisionReliability{
			TotalTimeout: "600s", PerTryTimeout: "300s", FirstByteTimeout: "20s", RetryCount: &retries,
			RetryOn: "reset", RetriableStatusCodes: []int32{429}, RetryAfterMax: "10s",
		},
		Fallback: &v1alpha1.DecisionFallback{
			Enabled: &enabled, MaxAttempts: 2, TotalTimeout: "900s", PerAttemptTimeout: "450s",
			RetryableStatusCodes: []int32{502, 503},
		},
	})
	require.NoError(t, err)

	one := 1
	assert.Equal(t, &config.DecisionReliability{
		TotalTimeout: "600s", PerTryTimeout: "300s", FirstByteTimeout: "20s", RetryCount: &one,
		RetryOn: "reset", RetriableStatusCodes: []int{429}, RetryAfterMax: "10s",
	}, decision.Reliability)
	assert.Equal(t, &fallback.FallbackOverride{
		Enabled: &enabled, MaxAttempts: 2, TotalTimeout: 900 * time.Second, PerAttemptTimeout: 450 * time.Second,
		RetryableStatusCodes: []int{502, 503},
	}, decision.Fallback)

	plain, err := (&CRDConverter{}).convertDecision(v1alpha1.Decision{Name: "plain"})
	require.NoError(t, err)
	assert.Nil(t, plain.Reliability, "a decision without the block keeps the provider model's")
	assert.Nil(t, plain.Fallback, "a decision without the block keeps the route's and the router's")
}

func TestConvertDecisionRefusesAFallbackTheRouterWouldRefuse(t *testing.T) {
	for name, spec := range map[string]v1alpha1.DecisionFallback{
		"duration": {PerAttemptTimeout: "soon"},
		"status":   {RetryableStatusCodes: []int32{700}},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := (&CRDConverter{}).convertDecision(v1alpha1.Decision{Name: "long_report", Fallback: &spec})
			require.Error(t, err)
			assert.Contains(t, err.Error(), "decision long_report")
		})
	}
}
