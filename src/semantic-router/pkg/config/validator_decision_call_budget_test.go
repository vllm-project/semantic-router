package config

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestValidateDecisionModelRefBudget(t *testing.T) {
	useReasoning := false
	decisionWithRefs := func(count int) Decision {
		refs := make([]ModelRef, 0, count)
		for i := 0; i < count; i++ {
			refs = append(refs, ModelRef{
				Model:                 fmt.Sprintf("model-%d", i),
				ModelReasoningControl: ModelReasoningControl{UseReasoning: &useReasoning},
			})
		}
		return Decision{Name: "wide-panel", ModelRefs: refs}
	}
	cfg := &RouterConfig{}

	// Exact boundary passes the full execution validator.
	require.NoError(t, validateDecisionExecution(cfg, decisionWithRefs(MaxUpstreamCallsPerRequest)))

	// One over the boundary is rejected with the observed count and the limit.
	err := validateDecisionExecution(cfg, decisionWithRefs(MaxUpstreamCallsPerRequest+1))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "decision 'wide-panel'")
	assert.Contains(t, err.Error(), "lists 33 candidate models")
	assert.Contains(t, err.Error(), "per-request upstream-call limit of 32")
}
