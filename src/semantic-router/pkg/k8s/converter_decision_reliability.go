package k8s

import (
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apis/vllm.ai/v1alpha1"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
)

// convertDecisionReliability copies a decision's reliability block. The
// router validates it, as any decision's, when it loads the configuration.
func convertDecisionReliability(reliability *v1alpha1.DecisionReliability) *config.DecisionReliability {
	if reliability == nil {
		return nil
	}
	converted := &config.DecisionReliability{
		TotalTimeout:         reliability.TotalTimeout,
		PerTryTimeout:        reliability.PerTryTimeout,
		IdleTimeout:          reliability.IdleTimeout,
		FirstByteTimeout:     reliability.FirstByteTimeout,
		RetryOn:              reliability.RetryOn,
		RetriableStatusCodes: intsOf(reliability.RetriableStatusCodes),
		RetryBackOffBase:     reliability.RetryBackOffBase,
		RetryBackOffMax:      reliability.RetryBackOffMax,
		RetryAfterMax:        reliability.RetryAfterMax,
	}
	if reliability.RetryCount != nil {
		count := int(*reliability.RetryCount)
		converted.RetryCount = &count
	}
	return converted
}

// convertDecisionFallback converts a decision's fallback block into the
// override the router lays over the route's and the router's policy.
func convertDecisionFallback(spec *v1alpha1.DecisionFallback) (*fallback.FallbackOverride, error) {
	if spec == nil {
		return nil, nil
	}
	total, err := optionalDuration(spec.TotalTimeout)
	if err != nil {
		return nil, fmt.Errorf("total_timeout: %w", err)
	}
	perAttempt, err := optionalDuration(spec.PerAttemptTimeout)
	if err != nil {
		return nil, fmt.Errorf("per_attempt_timeout: %w", err)
	}
	override := &fallback.FallbackOverride{
		MaxAttempts:          int(spec.MaxAttempts),
		TotalTimeout:         total,
		PerAttemptTimeout:    perAttempt,
		RetryableStatusCodes: intsOf(spec.RetryableStatusCodes),
	}
	if spec.Enabled != nil {
		enabled := *spec.Enabled
		override.Enabled = &enabled
	}
	return override, override.Validate()
}

func optionalDuration(raw string) (time.Duration, error) {
	if raw == "" {
		return 0, nil
	}
	return time.ParseDuration(raw)
}

func intsOf(values []int32) []int {
	if len(values) == 0 {
		return nil
	}
	out := make([]int, len(values))
	for i, value := range values {
		out[i] = int(value)
	}
	return out
}
